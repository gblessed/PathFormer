# Direct Model Inference Backend Design

**Goal:** Add a minimal inference backend for the direct Pathformer model that loads a scenario checkpoint and generates multipaths for a batch of 6D prompts.

## Scope

The backend will:
- accept a `scenario_name`
- resolve or accept a direct-model checkpoint path
- accept a list of prompts shaped as `[tx_x, tx_y, tx_z, rx_x, rx_y, rx_z]`
- run batched autoregressive generation up to `max_generate_steps`
- return generated paths, predicted interactions, and the path-count head output

The backend will not:
- build prompts from DeepMIMO scenario objects
- normalize or denormalize prompts
- add training or evaluation logic

## Model Assumptions

The implementation should match `multiscenario_direct_training.py`:
- model class: `PathDecoder`
- constructor: `PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True)`
- checkpoint key: `model_state_dict`
- generation helper: `generate_paths_no_env_batch`

## API Shape

Provide a small class with:
- constructor arguments for `scenario_name`, `checkpoint_path` or `checkpoint_dir`, and `device`
- a `generate(...)` method that accepts a batch of prompts and `max_generate_steps`

The `generate(...)` method should return one item per prompt with:
- `paths`: generated 7D path parameters
- `interactions`: generated 4D interaction labels
- `path_count`: scalar path-count prediction

## Validation

Keep validation simple:
- error on empty prompt vectors with wrong dimensionality
- error if the checkpoint path cannot be resolved
- allow an empty prompt list and return an empty result list
