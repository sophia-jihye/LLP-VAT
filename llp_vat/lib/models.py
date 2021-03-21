import torch.nn as nn
from torch.autograd import Function


class MLPModel(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(MLPModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(input_dim, 65))
        self.feature.add_module('f_relu1', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(65, 65))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(65, num_classes))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        feature = self.feature(input_data)
        out = self.class_classifier(feature)

        return out
