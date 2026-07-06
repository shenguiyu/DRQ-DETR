"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.core import YAMLConfig, yaml_utils
from engine.extre_module.torch_utils import check_cuda
from engine.misc import dist_utils
from engine.solver import TASKS


GREEN, RESET = "\033[92m", "\033[0m"


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    check_cuda()

    assert not all([args.tuning, args.resume]), (
        "Only one of scratch training, resume, or tuning is supported at a time"
    )

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({
        key: value
        for key, value in args.__dict__.items()
        if key not in ["update"] and value is not None
    })

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    cfg_str = json.dumps(cfg.__dict__, indent=4, ensure_ascii=False)
    print(GREEN + cfg_str + RESET)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit(cfg_str)

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="experiment seed")
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the YAML automatic mixed precision setting",
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="TensorBoard summary directory")
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+", help="override YAML values")
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")

    main(parser.parse_args())
