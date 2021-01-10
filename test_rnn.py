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
        num_telemetry_data_pts = params_model["num_telemetry_data_pts"]
        self.images_to_predict = params_model["images_to_predict"]
        
        baseModel = torchvision.models.resnet18(pretrained=pretrained).eval()
        
        num_features = baseModel.fc.in_features + num_telemetry_data_pts
        baseModel.fc = Identity()
        
        for param in baseModel.parameters():
            param.requires_grad = False
        
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size+1, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.rnn_hidden_size = rnn_hidden_size
        self.relu = nn.ReLU()
        
    def forward(self, img, tel):
        b_z, ts, c, h, w = img.shape
        ii = 0
        temp_out = []
        
        y = self.baseModel((img[:,ii]))
        y = torch.cat((y, tel[:,ii]), 1)
        
        out, (hn, cn) = self.rnn(y.unsqueeze(1))
        #out = out.squeeze(1)
        
        for ii in range(1, ts):
            y = self.baseModel((img[:,ii]))
            y = torch.cat((y, tel[:,ii]), 1)
            output, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            
            temp_out.append(output.squeeze(1))
            
        temp_out = torch.stack((temp_out),1)
        
        out = torch.cat((out, temp_out),1)[:,-2,:]
        
        tel = self.sigmoid(tel[:,-2,0].unsqueeze(1))
        
        out = torch.cat((out, tel), 1)
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sigmoid(out)
        
        return out
    
    @torch.no_grad()
    def predict(self, x, tel, hidden=None):        
        b_z, ts, c, h, w = x.shape
        ii = 0
        
        self.baseModel.eval()
        
        y = self.baseModel((x[:,ii]))
        
        y = torch.cat((y, tel), 1)
        
        if(hidden==None):
            out, (hn, cn) = self.rnn(y.unsqueeze(1))
        else:
            out, (hn, cn) = self.rnn(y.unsqueeze(1), hidden)

        tel = self.sigmoid(tel[:,0].unsqueeze(0).unsqueeze(1))
        
        out = torch.cat((out, tel), 2)
        
        out = self.fc1(out) 
        out = self.sigmoid(out)
        
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