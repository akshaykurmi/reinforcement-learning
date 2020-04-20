import argparse
import os
import shutil

from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=["train", "test"], required=True, help="Train or test the agent?")
parser.add_argument('--ckpt_dir', required=True, help="Name of checkpoint directory")
parser.add_argument('--log_dir', required=True, help="Name of log directory")
parser.add_argument("--run_id", default="run_1", help="Run ID")
parser.add_argument("--overwrite_results", default=False, action="store_true")
args = parser.parse_args()

args.base_dir = os.path.dirname(__file__)
args.ckpt_dir = os.path.join(args.base_dir, args.ckpt_dir, args.run_id)
args.log_dir = os.path.join(args.log_dir, args.log_dir, args.run_id)

args.max_action_steps = 10000000
args.train_steps = 1
args.batch_size = 32
args.gamma = 0.95
args.update_dqn_target_steps = 1
args.save_steps = 1

agent = Agent()

if args.overwrite_results:
    if os.path.exists(args.ckpt_dir):
        shutil.rmtree(args.ckpt_dir)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

if args.mode == "train":
    agent.train(args)

if args.mode == "test":
    agent.test()
