from dataclasses import dataclass

import torch
import transformers
from transformers import HfArgumentParser


@dataclass
class Args:
    checkpoint_path: str = (
        "mistral-gemma-test"
    )
    model_class: str = "AutoModelForCausalLM"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    #flax_model = getattr(transformers, "Flax" + args.model_class).from_pretrained(args.checkpoint_path)
    # only way I managed to get it to work: save as a single flax model by increasing max_shard_size even more.
    #flax_model.save_pretrained(args.checkpoint_path, max_shard_size="30GB")

    #del flax_model
    pt_model = getattr(transformers, args.model_class).from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16,
                                                                       from_flax=True)
    pt_model.save_pretrained(args.checkpoint_path)
