###################################################################################################
# In this file, we define the ExperimentManager class, which is used to manage the experiment
# configuration, logging, and checkpointing. The ExperimentManager class is used in the main
# script to set up the experiment, log the configuration, and save the checkpoints.
#
# The ExperimentManager class has the following methods:
# - get_basic_arg_parser: Returns an argument parser with the basic options for the project.
# - _setup_options: Parses the input arguments and sets the options for the project.
# - _setup_dirs: Sets up the log directory and checkpoint directory for the project.
# - _set_up_logger: Sets up the logger for the project.
# - _setup_seed: Sets the random seed for reproducibility.
# - _setup_torch: Sets up the PyTorch environment.
# - _export_arguments: Saves the experiment configuration to a file.
# - _setup: Sets up the experiment by calling the above methods.
# - get_opt: Returns the experiment options.
# - get_logger: Returns the logger for the project.
# - get_log_dir: Returns the log directory for the project.
# - get_checkpoint_dir: Returns the checkpoint directory for the project.
###################################################################################################

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch


class ExperimentManager(object):
    def __init__(self):
        # Options for this project
        self._opt = None
        # Logger for project
        self._logger = None
        # Running result log dir for this project
        self._log_dir = None
        # Checkpoint dir for this project
        self._checkpoint_dir = None

        # Set up the experiment manager
        self._setup()

    def get_basic_arg_parser(self):
        parser = argparse.ArgumentParser()
        # Basic options
        parser.add_argument('--gene_map', type=str, required=True, help="gene map tif file, with shape (n, m, k), k is the number of genes")
        parser.add_argument('--nuclei_mask', type=str, required=True, help="paired nuclei mask tif file, with shape (n, m)")
        parser.add_argument('--basis', type=str, required=True, help="basis tif file, with shape (c, k), c is the number of cell types")
        parser.add_argument('--log_dir', type=str, default="./Log", help="run dir for logging, default is ./Log")
        parser.add_argument('--normalized_total', type=int, default=1e3, help="normalized total for gene map")
        parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")

        # Model options
        parser.add_argument('--patch_size', type=int, default=64, help="patch size for training and prediction")
        parser.add_argument('--deconv_emb_dim', type=int, default=128, help="embedding dimension for deconvolution")
        parser.add_argument('--fpn_level', type=int, default=-1, help="fpn level for deconvolution")

        # Training options
        parser.add_argument('--epochs', type=int, default=40, help="number of epochs for training")
        parser.add_argument('--lr_bb', type=float, default=1e-3, help="learning rate for backbone")
        parser.add_argument('--lr_fpn', type=float, default=1e-3, help="learning rate for fpn")
        parser.add_argument('--lr_deconv', type=float, default=1e-2, help="learning rate for deconv")
        parser.add_argument('--lr_decoder', type=float, default=1e-3, help="learning rate for decoder")
        parser.add_argument('--recon_loss_weight', type=float, default=0.1, help="reconstruction loss weight")
        parser.add_argument('--gradient_clip', type=float, default=10.0, help="gradient clip value")

        # Environment options
        parser.add_argument('--gpu', type=str, default="0", help="gpu device id")
        parser.add_argument('--seed', type=int, default=0, help="random seed")

        # Test options
        parser.add_argument('--load_path', type=str, default=None, help="load model path for testing")

        return parser

    def _setup_options(self):
        parser = self.get_basic_arg_parser()
        self._opt, _ = parser.parse_known_args()

    def _setup_dirs(self):
        opt = self._opt
        self._log_dir = opt.log_dir
        if os.path.exists(self._log_dir):
            print(f"Log dir exists: {self._log_dir}, please choose an option:")
            op = input("d (delete) / n (new) / q (quit): ")
            if op == 'd':
                shutil.rmtree(opt.log_dir, ignore_errors=True)
                print("Old files deleted.")
            elif op == 'n':
                self._log_dir = opt.log_dir + f"_new_{int(time.time())}"
            else:
                raise OSError("Quit without changes.")
        os.makedirs(self._log_dir, exist_ok=True)
        print(f"Log dir: {self._log_dir}")
        # Checkpoint dir
        self._checkpoint_dir = os.path.join(self._log_dir, "checkpoint")
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {self._checkpoint_dir}")

    def _set_up_logger(self, level=logging.DEBUG, name="UCS"):
        """
        Setting self._logger
        """
        self._logger = logging.getLogger(name=name)
        self._logger.propagate = False
        self._logger.setLevel(level)
        # Stdout handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
        # Log file handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        filename = "log.log"
        path = os.path.join(self._log_dir, filename)
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def _setup_seed(self):
        random.seed(self._opt.seed)
        np.random.seed(self._opt.seed)
        torch.manual_seed(self._opt.seed)
        torch.cuda.manual_seed_all(self._opt.seed)

    def _setup_torch(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self._opt.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def _export_arguments(self):
        """
        Save self._opt
        """
        opt = self._opt
        self._logger.info(f"Opts: {opt}")
        with open(os.path.join(self._log_dir, 'argv.txt'), 'w') as f:
            print(sys.argv, file=f)

    def _setup(self):
        self._setup_options()
        self._setup_dirs()
        self._set_up_logger()
        self._setup_seed()
        self._setup_torch()
        self._export_arguments()

    def get_opt(self):
        return self._opt

    def get_logger(self):
        return self._logger

    def get_log_dir(self):
        return self._log_dir

    def get_checkpoint_dir(self):
        return self._checkpoint_dir

if __name__ == "__main__":
    # Test the ExperimentManager
    manager = ExperimentManager()
    opt = manager.get_opt()
    logger = manager.get_logger()
    log_dir = manager.get_log_dir()
    checkpoint_dir = manager.get_checkpoint_dir()
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Options: {opt}")
    logger.info("ExperimentManager test passed.")
