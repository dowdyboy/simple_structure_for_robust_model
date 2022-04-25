from torch.utils.tensorboard import SummaryWriter
from .log import log

_tf_writer = None


def tf_board_init(*args, **kv):
    global _tf_writer
    if _tf_writer is None:
        _tf_writer = SummaryWriter(*args, **kv)

    else:
        log('tf writer already inited. this call do nothing')


def add_scalar(*args, **kv):
    global _tf_writer
    _tf_writer.add_scalar(*args, **kv)
