# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from comet_ml import Experiment
from trainer import Trainer
from options import MonodepthOptions
import torch
import random
options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    #torch.backends.cudnn.deterministic = True
    #random.seed(2)
    #torch.manual_seed(2)
    #torch.cuda.manual_seed(2)
    #experiment = Experiment(api_key="l6NAe3ZOaMzGNsrPmy78yRnEv", project_name="depth2", workspace="tehad")
    trainer = Trainer(opts)
    trainer.train()
