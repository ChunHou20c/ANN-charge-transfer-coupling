from config import LEARNING_RATE , DECAY_RATE
import math

def lr_exp_decay(epoch, lr):
    return LEARNING_RATE * math.exp(-DECAY_RATE*epoch)
