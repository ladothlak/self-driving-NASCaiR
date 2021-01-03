from torch import nn
import torchvision
import torch

class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = torchvision.models.resnet18(pretrained=pretrained).eval()
        
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        
        for param in baseModel.parameters():
            param.requires_grad = False
        
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.rnn_hidden_size = rnn_hidden_size
        self.relu = nn.ReLU()
        
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        
        y = self.baseModel((x[:,ii]))
        
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out)
        
        sign = torch.sign(out)
        out = self.relu(sign)
        
        return out 
    
    def predict(self, x, hidden):
        torch.no_grad()
        
        b_z, ts, c, h, w = x.shape
        ii = 0
        
        self.baseModel.eval()
        
        y = self.baseModel((x[:,ii]))
        
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))

        out = self.fc1(out) 
        
        sign = torch.sign(out)
        out = self.relu(sign)
        
        return out, (hn, cn)
    
    def init_hidden(self):
        
        h0 = torch.zeros(2, 1, self.rnn_hidden_size).cuda()
        c0 = torch.zeros(2, 1, self.rnn_hidden_size).cuda()
        
        return (h0, c0)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x