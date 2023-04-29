import os
import time

import argparse
import yaml
from loguru import logger


def parse_common_args(parser):
    parser.add_argument('--model_name', type=str, default='base_model', help='which model to train or eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--yaml", type=str, default=None, help="yaml configuration to update args")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="which device to run model")
    parser.add_argument("--log_to_file", action="store_true", help="print log to file")
    parser.add_argument("--project_root_path", type=str, default="./", help="the root path of the all output which to save info")
    parser.add_argument('--param_path', type=str, default=None, help='which model parameter to initialize the model')
    return parser


def add_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--criterion', type=str, default="cross_entropy", help='loss function')
    parser.add_argument('--dataset', type=str, default="minist_dataset", help='dataset')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    # parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_by_step', type=int, default=4, help="The step to display train information.")
    return parser


def add_eval_args(parser):
    parser.add_argument('--result_dir', type=str, required=True, help='The path to save eval results')
    parser = parse_common_args(parser)
    return parser


def update_args_by_yaml(args):
    yaml_path = args.yaml
    if yaml_path is not None:
        logger.debug(yaml_path)
        if not os.path.exists(yaml_path):
            raise Exception("The path {} does not exists!".format(yaml_path))
        with open(yaml_path, encoding="utf-8") as f:
            opt = yaml.load(f, Loader = yaml.FullLoader)

        args = vars(args)
        args.update(opt)
    return args


def get_formated_args(args):
    infos = [""]
    if isinstance(args, dict):
        for key, val in args.items():
            infos.append(("{:>20}: {}".format(key, val)))
    else:
        for key, val in vars(args).items():
            infos.append(("{:>20}: {}".format(key, val)))

    return "\n".join(infos)


if __name__ == '__main__':
    # train_args = get_train_args()
    # test_args = get_test_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--aa', required=True, help="file path")

    subparsers = parser.add_subparsers(help='commands')
    subparsers.default = "read"
    

    read_parser = subparsers.add_parser(name='read', help='read file')
    read_parser.add_argument('--path', help="file path")
    read_parser.add_argument('--length', help="file length")
    read_parser.add_argument('--key', help="file key")

    write_parser = subparsers.add_parser(name='write', help='write file')
    write_parser.add_argument('--path', help="file path")
    write_parser.add_argument('--encoding', help="file encoding")

    args = parser.parse_args()
    
