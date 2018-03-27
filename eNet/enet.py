import tensorflow as tf
import numpy as np
import tool

def enet(x, n_classes, is_pretrain=False):
    x = tool.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tool.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tool.conv('conv3_1',x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.conv('conv3_2',x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tool.pool('pool3',x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    x = tool.batch_norm(x)

    x = tool.FC_layer('fc4', x, out_nodes=4096)
    x = tool.batch_norm(x)
    x = tool.FC_layer('fc5', x, out_nodes=4096)
    x = tool.batch_norm(x)
    x = tool.FC_layer('fc6', x, out_nodes=n_classes)

    return x
