# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:46:28 2024

@author: reinh
"""
import tensorflow as tf
import numpy as np
import math
    
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, warmup_target, alpha=0.0):
        super(WarmUpCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target
        self.alpha = alpha
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return (self.warmup_target / self.warmup_steps) * (step + 1)
        else:
            cosine_decay = 0.5 * (1 + tf.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.warmup_target * decayed
