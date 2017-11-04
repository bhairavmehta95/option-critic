import numpy as np
import tensorflow as tf

from model import MLP3D, Model

def clip_grads(grads, clip, clip_type):
  if clip > 0.1: 
    if clip_type == "norm":
      grads = [norm_constraint(p, clip) if p.ndim > 1 else T.clip(p, -clip, clip) for p in grads]
    elif clip_type == "global":
      norm = T.sqrt(T.sum([T.sum(T.sqr(g)) for g in grads])*2) + 1e-7
      scale = clip * T.min([1/norm,1./clip]).astype("float32")
      grads = [g*scale for g in grads]
  return grads