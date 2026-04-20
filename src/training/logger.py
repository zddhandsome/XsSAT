"""
SATformer 风格的日志记录器
与 SATformer 项目保持一致的日志格式
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import torch

USE_TENSORBOARD = True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        import tensorboardX
        SummaryWriter = tensorboardX.SummaryWriter
    except ImportError:
        USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, save_dir, debug_dir=None, tmp_dir=None, config=None):
        """
        创建日志记录器

        Args:
            save_dir: 保存目录
            debug_dir: 调试目录（可选）
            tmp_dir: 临时目录（可选）
            config: 配置对象或字典（可选）
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        if tmp_dir and not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        # 保存配置到 opt.txt
        file_name = os.path.join(save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            if config is not None:
                if isinstance(config, dict):
                    args = config
                else:
                    args = dict((name, getattr(config, name)) for name in dir(config)
                                if not name.startswith('_'))
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = save_dir + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.writer = None
        self.log = open(log_dir + '/log.txt', 'w')
        self.log_dir = log_dir
        try:
            os.system('cp {}/opt.txt {}/'.format(save_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        """写入日志"""
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        """关闭日志"""
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """记录标量到 TensorBoard"""
        if USE_TENSORBOARD and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
