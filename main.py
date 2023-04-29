import argparse
from loguru import logger

from options import *
import trainer
from tqdm import tqdm


def train(args):
    logger.info("*************** train ***************")
    update_args_by_yaml(args)

    #########################################################################################
    # call the implementation of train phase
    #########################################################################################
    trainer.main(args)


def eval(args):
    logger.info("*************** eval ***************")
    update_args_by_yaml(args)
    logger.info(get_formated_args(args))

    #########################################################################################
    # call the implementation of eval phase
    #########################################################################################

    pass


if __name__ == "__main__":
    # delete_blank_checkpoint("./checkpoints")
    # logger.remove()
    # logger.add(wrap_tqdm_write)
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(required=True)

    # ######################################## train subparser #######################################
    train_parser = subparser.add_parser(name='train', help='train model')
    train_parser.set_defaults(func=train)
    add_train_args(train_parser)

    # ######################################## eval subparser ########################################
    eval_parser = subparser.add_parser(name="eval", help="eval model")
    eval_parser.set_defaults(func=eval)
    add_eval_args(eval_parser)

    #################################################################################################
    args = parser.parse_args()
    args.func(args)