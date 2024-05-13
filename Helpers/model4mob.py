"""
reference
https://tutorials.pytorch.kr/advanced/cpp_export.html

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from torch.utils.mobile_optimizer import optimize_for_mobile

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
        
        self.get_output_shape()
        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes))
        self.dequant = torch.quantization.DeQuantStub()
        
    def get_output_shape(self):
        tmp = torch.rand((1,1,28,625))
        with torch.no_grad():
            output = self.convnet(tmp)
            self.n_outputs = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]


    def forward(self, x):
        x = self.quant(x)  
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output= self.clf(output) 
        output = self.dequant(output)
        output = F.softmax(output, dim=1)         
        return output

if __name__ == '__main__':

    ############ 양자화 위한 모델 생성
    model = EEGNet4(3, 28, 625, True) # n_classes, n_channel, n_timewindow
    """    
    model.eval
    print(model)

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    model = torch.quantization.convert(model, inplace=True)
    print(model)
    """

    ############# 학습 시작
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 훈련 데이터와 레이블 생성 
    train_data = torch.rand(16, 1, 28, 625)  # 16개의 훈련 데이터 예제
    train_labels = torch.randint(0, 3, (16,))  # 16개의 레이블 (0, 1, 또는 2 중 하나)

    # 추론 데이터 생성 
    data = torch.rand(1,1,28,625)  

    # 모델 훈련
    num_epochs = 10  # 에폭 수 설정

    for epoch in range(num_epochs):
        model.train()  # 모델을 훈련 모드로 설정
        optimizer.zero_grad()  # 그래디언트 초기화

        # 순전파
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    

    # 모델을 평가 모드로 설정
    model.eval()

    # 추론 수행
    with torch.no_grad():
        test_data = data 
        predictions = model(test_data)
        print("Predictions:", predictions)      

    #### 양자화 수행
    model.eval
    print(model)

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    model = torch.quantization.convert(model, inplace=True)
    print(model)

    #### 테스트
    model.eval()

    # 추론 수행
    with torch.no_grad():
        test_data = data 
        predictions = model(test_data)
        print("Predictions:", predictions)   

    torchscript_model = torch.jit.trace(model, data)
    
    # Export mobile interpreter version model (compatible with mobile interpreter)
    optimized_scripted_module = optimize_for_mobile(torchscript_model)
    optimized_scripted_module._save_for_lite_interpreter("/opt/workspace/Seohyeon/SOSO_App/scripted_model.ptl")
