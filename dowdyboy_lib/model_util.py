

def frozen_module(module):
    for key, value in module.named_parameters():  # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
        value.requires_grad = False


def unfrozen_module(module):
    for key, value in module.named_parameters():
        value.requires_grad = True

