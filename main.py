# /*
#  * Copyright (c) 2025 zhitian111
#  * Released under the MIT license. See LICENSE for details.
#  */
from utility.args import parse_args
from utility.logger import setup_logger
from train import train
from eval import eval
args = parse_args()
logger_all = setup_logger(args.exp_name, args.log_dir, "all.log")
logger_train = setup_logger(args.exp_name, args.log_dir, "train.log")
logger_eval = setup_logger(args.exp_name, args.log_dir, "eval.log")


if __name__ == "__main__":
    logger_all.info("开始运行...")
    if args.mode == "train":
        logger_train.info("<UNK>...")
        train(args, logger_train)
    elif args.mode == "eval":
        logger_eval.info("<UNK>...")
        eval(args, logger_eval)
    logger_all.info("运行结束...")