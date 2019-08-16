from cnn.resnet.net import ResNetLayer, ResNetSection, ResNet
from cnn.se_resnet.net import SEResNetBlock
from cnn.origin.train import Train
from cnn.origin.batch import Batch, TestBatch

net = ResNet().build(ResNetLayer(), SEResNetBlock(), ResNetSection())\
              .set_scope('se_res_net_50').set_layer_num(50)
Train().build(net, Batch(), TestBatch()).exec()
