"""
Simple EEGNet
based on "~" paper
"""
import torch
import torch.nn as nn

class EEGNet4(nn.Module): # 4 temporal, 2 spatial per temporal
    def __init__(self, args, track_running=True):
        super(EEGNet4, self).__init__()
        self.n_classes = num_classes

        num_classes = args.n_classes
        input_ch = args.n_channels
        input_time = args.freq_time
        freq = input_time

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, freq//2), stride=1, bias=False, padding=(1 , freq//4)),
            nn.BatchNorm2d(4, track_running_stats=track_running),
            nn.Conv2d(4, 8, kernel_size=(input_ch, 1), stride=1, groups=4),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(8, 8, kernel_size=(1,freq//4),padding=(0,freq//4), groups=8),
            nn.Conv2d(8, 8, kernel_size=(1,1)),
            nn.BatchNorm2d(8, track_running_stats=track_running),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25),
            )
    
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output=self.clf(output) 
        return output

if __name__ == '__main__':
    model = EEGNet4(2, 30, 384, 10, True) # n_classes, n_channel, n_timewindow
    # pred = model(torch.zeros(50, 1, 20, 250))
    print(model)
    from pytorch_model_summary import summary

    print(summary(model, torch.rand((1, 1, 30, 384)), show_input=True))
    # model input = torch.rand((1,1,32,600))
    # batch size, channel, eeg electrodes, time window 