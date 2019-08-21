from cnn.resnet.net import ResNetLayer, ResNetBlock, ResNetSection, ResNet
from cnn.origin.train import Train
from cnn.origin.batch import Batch, TestBatch

net = ResNet().build(ResNetLayer(), ResNetBlock(), ResNetSection())\
              .set_scope('res_net_18_1').set_layer_num(18)
Train().build(net, Batch(), TestBatch()).exec()
