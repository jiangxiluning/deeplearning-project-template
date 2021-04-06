import argparse
import pathlib

from pytorch_lightning import Trainer
import anyconfig
import easydict

from project.tools.utils import add_argparse_args
from project.tools.train import train as _train



def train(args: argparse.Namespace):
    config = anyconfig.load(args.config, ac_parser='yaml')
    config = easydict.EasyDict(config)

    command_opts = vars(args)
    _train(config=config,
           command_opts=command_opts)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main')

    sub_parsers = parser.add_subparsers(dest='name')
    train_parser = sub_parsers.add_parser(name='train', description='training utility')
    train_parser.add_argument('config',
                              help='config file')
    train_parser = add_argparse_args(Trainer, train_parser)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)