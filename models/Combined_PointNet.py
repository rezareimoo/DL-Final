import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Transform import Transform

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fca = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bna = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = torch.tanh(self.bn1(self.fc1(xb)))
        xb = torch.tanh(self.bn2(self.fc2(xb)))
        xb = torch.tanh(self.bna(self.dropout(self.fca(xb))))
        xb = self.fc3(xb)
        # print('after fc3', xb)
        xb = nn.AdaptiveMaxPool1d(xb.size(1))(xb)
        # print('after adaptive', xb)
        output = nn.Flatten()(xb)
        # print('after flatten', output)
        # print('after softmax', self.logsoftmax(output))
        return self.logsoftmax(output), matrix3x3, matrix64x64
    

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.MultiLabelMarginLoss()
    bs=outputs.size(0)
    neg_1s = torch.ones(outputs.size(0), outputs.size(1)-1) * -1
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        neg_1s = neg_1s.cuda()
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    labels_reshape = torch.cat((labels[:, None], neg_1s), dim=1)
    labels_reshape = labels_reshape.to(dtype=torch.long)
    return criterion(outputs, labels_reshape) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)