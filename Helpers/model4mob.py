"""
Simple EEGNet
based on "~" paper
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.mobile_optimizer import optimize_for_mobile

from pytorch_model_summary import summary

###########################
# 모바일 올리기 위한 양자화 #
###########################

class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, num_classes, input_ch, input_time, track_running=True):
        super(EEGNet4, self).__init__()

        self.quant = torch.quantization.QuantStub()

        self.n_classes = num_classes
        freq = input_time

        self.convnet = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(1, 4, kernel_size=(1, freq//2), stride=1, bias=False, padding=(1 , freq//4)).to(dtype=torch.float)),
            ('BN_1',nn.BatchNorm2d(4, track_running_stats=track_running).to(dtype=torch.float)),
            ('conv_2', nn.Conv2d(4, 8, kernel_size=(input_ch, 1), stride=1, groups=4).to(dtype=torch.float)),
            ('BN_2', nn.BatchNorm2d(8, track_running_stats=track_running).to(dtype=torch.float)),
            ('elu_1', nn.ELU()),
            ('avgPool_1',nn.AvgPool2d(kernel_size=(1,4)).to(dtype=torch.float)),
            ('dropOut_1', nn.Dropout(p=0.25).to(dtype=torch.float)),
            ('conv_3', nn.Conv2d(8, 8, kernel_size=(1,freq//4),padding=(0,freq//4), groups=8).to(dtype=torch.float)),
            ('conv_4', nn.Conv2d(8, 8, kernel_size=(1,1)).to(dtype=torch.float)),
            ('BN_3', nn.BatchNorm2d(8, track_running_stats=track_running).to(dtype=torch.float)),
            ('elu_2', nn.ELU()),
            ('avgPool_2', nn.AvgPool2d(kernel_size=(1, 8)).to(dtype=torch.float)),
            ('dropOut_2', nn.Dropout(p=0.25).to(dtype=torch.float)),
            ]))

        self.clf = nn.Sequential(nn.Linear(17500, self.n_classes))
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        # x = x[0].unsqueeze(dim=1).permute(0,1,2,3)  # eeg
        x = self.quant(x)  # 양자화
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output= self.clf(output) 
        output = self.dequant(output)
        return output

if __name__ == '__main__':
    model = EEGNet4(3, 28, 625, True) # n_classes, n_channel, n_timewindow
    # pred = model(torch.zeros(50, 1, 20, 250))
    model.eval
    print(model)

    """
    # fuse -> 정확도 증가
    modules_to_fuse = []
    modules_to_fuse.append(['convnet.conv_1', 'convnet.BN_1', 'convnet.conv_2', 
                            'convnet.BN_2', 'convnet.elu_1', 'convnet.avgPool_1', 
                            'convnet.dropOut_1', 'convnet.conv_3', 'convnet.conv_4',
                            'convnet.BN_3', 'convnet.elu_2','convnet.avgPool_2','convnet.dropOut_2'])
    modules_to_fuse.append(['clf'])
    torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    """

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    model = torch.quantization.convert(model, inplace=True)

    print(model)

    torchscript_model = torch.jit.script(model)

    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "/opt/workspace/Seohyeon/SOSO_App/res/model.pt")

    # print(summary(model, torch.rand((16, 1, 28, 625)), show_input=True))
    # model input = torch.rand((1,1,32,600))
    # batch size, channel, eeg electrodes, time window 