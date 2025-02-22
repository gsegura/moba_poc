


__version__ = "0.1.0""""Deep Research POC using MoBA and KV Cache."""

from moba_research.attention import MoBAConfig, moba_attn_varlen
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from functools import partial

from moba_research.attention_wrapper import moba_layer


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)