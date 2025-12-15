import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model"
    )

    # data
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    # training
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=2.7,
    )
    parser.add_argument(
    "--method",
        type=str,
        default="3DCNN"
    )

    # system
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./results/" + time.strftime("%Y%m%d-%H%M%S") + "/models/",
        help="directory to save model"
    )

    # log
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/log_" + time.strftime("%Y%m%d-%H%M%S"),
    )

    # run mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
    )


    # eval
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to the checkpoint file"
    )
    return parser.parse_args()