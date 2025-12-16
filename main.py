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
logger_gui = setup_logger(args.exp_name, args.log_dir, "gui.log")

def gui_main():
    from gui.main_gui import gui_main
    gui_main()

if __name__ == "__main__":
    random_seed = args.seed
    import torch
    import numpy as np
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    logger_all.info("开始运行...")
    if args.mode == "train":
        logger_train.info("<UNK>...")
        train(args, logger_train)
    elif args.mode == "eval":
        logger_eval.info("<UNK>...")
        eval(args, logger_eval)
    elif args.mode == "gui":
        logger_gui.info("<UNK>...")
        gui_main()

    logger_all.info("运行结束...")

