# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np

def sigmoid_rampup(current, rampup_length,max_coef=1.):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    """Modified version from https://github.com/vikasverma1077/GraphMix/blob/master/semisupervised/codes/ramps.py"""
    if rampup_length == 0:
        return max_coef
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))*max_coef

def cosine_rampdown(current, rampdown_length,max_coef=1.):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi *current / rampdown_length) + 1))*max_coef