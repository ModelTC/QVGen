import torch
from accelerate import Accelerator


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_transformer_parameters: int = 0
    num_trainable_w_quantization_parameters: int = 0
    num_trainable_act_quantization_parameters: int = 0
    num_trainable_round_quantization_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None
    num_update_steps_per_epoch: int = None

    # Hub state
    repo_id: str = None

    # Artifacts state
    output_dir: str = None
