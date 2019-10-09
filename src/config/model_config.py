#!/usr/bin/env python
from yacs.config import CfgNode as CN


_C = CN()
_C.MODEL = CN()
_C.MODEL.ENC_HIDDEN_SZ = 300
_C.MODEL.CTX_HIDDEN_SZ = 200
_C.MODEL.CTX_IN_SZ = 100
_C.MODEL.DEC_HIDDEN_SZ = 300
_C.MODEL.EMBED_SZ = 5
_C.MODEL.NUM_LAYERS = 2
_C.MODEL.NUM_DIRECTIONS = 1

cfg = _C  # users can `from config import cfg`
