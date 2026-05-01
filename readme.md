# Polaris

Official code of Polaris accepted in ICML 2026.

## Project Layout

```text
Polaris/
+-- readme.md
+-- image_taxo.py
+-- src/
    +-- main.py          # training entry point
    +-- main_pred.py     # checkpoint evaluation / prediction entry point
    +-- exp.py           # experiment driver: train, evaluate, visualize
    +-- model.py         # Polaris model
    +-- data.py          # PyTorch datasets and dataloaders
    +-- pre_process.py   # taxonomy preprocessing utilities
    +-- utils.py         # metrics and plotting utilities
    +-- svgd.py          # SVGD kernels
    +-- vmf.py           # vMF regularization
    +-- manifolds/       # sphere manifold operations
```

The scripts use paths such as `../data`, `../result`, and `../results`. Run commands from `src/` unless you intentionally adapt these paths.

## Environment

The project expects a Python environment with PyTorch, Transformers, NetworkX, scikit-learn, pandas, matplotlib, seaborn, tqdm, Weights & Biases, and OpenCLIP for image-taxonomy experiments.

Example setup:

```bash
conda create -n polaris python=3.12
conda activate polaris
pip install torch transformers networkx scikit-learn pandas matplotlib seaborn tqdm wandb open_clip_torch pillow
```

The default code loads local pretrained models from:

```text
/home/models/bert-base-uncased
/home/models/CLIP-ViT-H-14-laion2B-s32B-b79K
```

Update those paths in `src/model.py` and `src/exp.py` if your pretrained backbones are stored elsewhere.

## Data Layout

Place raw and processed taxonomy files under `data/<dataset>/` relative to the repository root:

```text
data/
+-- <dataset>/
    +-- <dataset>.terms
    +-- <dataset>.taxo
    +-- <dataset>_train.terms
    +-- <dataset>_val.terms
    +-- <dataset>_test.terms
    +-- processed/
        +-- taxonomy_data_<expID><negsamples><seed>_.pkl
```

If the processed pickle is missing, `main.py` calls the corresponding preprocessing routine automatically:

```text
create_mag_data      for computer_science, psychology, mesh, wordnet_verb, semeval_food
create_image_data    for birds
create_data          for science, environment, wordnet, and other text taxonomies
```

## Training with `main.py`

From the repository root:

```bash
cd src
python main.py \
  --dataset environment \
  --exp_name camera_ready_environment \
  --expID 510 \
  --seed 20 \
  --gpu_id 0 \
  --wandb 0
```

`main.py` parses command-line arguments and then applies dataset-specific defaults in the `if __name__ == '__main__'` block. For example, `science`, `environment`, and `wordnet` are treated as single-parent settings, while datasets such as `mesh`, `wordnet_verb`, `semeval_food`, and `birds` receive dataset-specific epoch, batch-size, negative-sampling, and embedding-size settings.

Important: the current camera-ready script also overwrites several parsed values before training, including `gpu_id`, `expID`, `batch_size`, `accumulation_steps`, `geometric_weight`, `c`, `vmf_margin`, and `svgd_weight`. Edit that block in `src/main.py` when you want a command-line value to take precedence over the provided paper defaults.

Training outputs are written to:

```text
../result/<dataset>/train/<exp_name>/
../result/<dataset>/model/<exp_name>/
../final_result/<dataset>/<exp_name>/
../gradients/<dataset>/<exp_name>/
```

## Prediction with `main_pred.py`

Use `main_pred.py` to load a trained checkpoint and run level-wise prediction on the test split:

```bash
cd src
python main_pred.py \
  --dataset environment \
  --expID 510 \
  --negsamples 50 \
  --seed 20 \
  --embed_size 32 \
  --batch_size 128 \
  --gpu_id 0 \
  --model_path ../final_result/environment/camera_ready_environment/experiment_<setting>.pt
```

The prediction script initializes `Experiments(args)` and calls:

```python
exp.level_wise_prediction(tag='test', path=args.model_path)
```

Evaluation summaries and qualitative plots are written under `../results/<dataset>/`.

`main_pred.py` shares most arguments with `main.py`, but it is intentionally evaluation-oriented: it sets CUDA inference mode directly and requires `--model_path` for loading a checkpoint. Pass the same `dataset`, `expID`, `negsamples`, `seed`, `embed_size`, and model-related arguments that were used during training.

## Key Arguments

| Argument | Default in `main.py` | Description |
| --- | ---: | --- |
| `--dataset` | `environment` | Dataset name under `../data/<dataset>/`. |
| `--model` | `bert` | Text backbone identifier used by the model. Supports the local BERT path and Snowflake branch used in code. |
| `--pre_train` | `bert` | Metadata label for the pretrained model setting. |
| `--hidden` | `64` | Hidden dimension of projection MLPs. |
| `--embed_size` | `8` | Spherical embedding dimension before dataset-specific overrides. |
| `--dropout` | `0.4` | Dropout hyperparameter retained for experiment configuration. |
| `--padmaxlen` | `30` | Maximum token length for text inputs. |
| `--matrixsize` | `768` | Backbone embedding size metadata. |
| `--negsamples` | `20` | Number of negative parents sampled per node. |
| `--epochs` | `55` | Number of training epochs before dataset-specific overrides. |
| `--batch_size` | `512` | Training batch size before dataset-specific overrides. |
| `--lr` | `9e-5` | Learning rate for model parameters. |
| `--lr_proj` | `1e-3` | Projection learning-rate metadata retained in the config. |
| `--eps` | `1e-8` | Adam/RiemannianAdam numerical epsilon. |
| `--optim` | `adamw` | Optimizer label. |
| `--accumulation_steps` | `5` | Gradient-accumulation configuration. |
| `--beta` | `0.5` | Margin for negative taxonomy samples. |
| `--vmf_margin` | `0.5` | Margin used by the vMF regularization loss. |
| `--c` | `0.7` | Welsch loss scale parameter. |
| `--geometric_weight` | `0.5` | Weight on the angular taxonomy loss. |
| `--probabilistic_weight` | `0.5` | Weight metadata for probabilistic loss components. |
| `--svgd_weight` | required/none | Weight on the SVGD regularization term. Set explicitly for training runs. |
| `--kappa_align` | `2.5` | vMF/SVGD alignment concentration. |
| `--kappa_repel` | `4.5` | vMF/SVGD repulsion concentration. |
| `--kernel_setting` | `vmf_theta` | SVGD kernel setting: code paths include `radial`, `vmf`, `imq`, and `vmf_theta`. |
| `--experiment_setting` | `standard` | SVGD experiment mode; `constant_svgd` uses embeddings as their own target particles. |
| `--learn_mu` | `1` | Whether to learn vMF mean parameters. |
| `--learn_kappa` | `1` | Whether to learn vMF concentration parameters. |
| `--detach_svgd` | `0` | Disable SVGD contribution when set to `1`. |
| `--implement_rectangular_opt` | `False` | Use rectangular polar-coordinate optimization instead of sphere optimization. |
| `--potential_strength` | `0.5` | Strength of the orbital radial potential used in level-wise prediction. |
| `--is_multi_parent` | `True` | Whether the taxonomy has multiple valid parents per query. Dataset defaults may override this. |
| `--cuda` | `True` | Enable CUDA when available. |
| `--gpu_id` | `2` | CUDA device index before dataset-specific overrides. |
| `--seed` | `20` | Random seed for Python, NumPy, and PyTorch. |
| `--expID` | `8` | Experiment/preprocessing identifier used in processed pickle names. |
| `--exp_name` | `experiment_name` | Name used for output directories. |
| `--wandb` | `1` | Enable Weights & Biases logging. Use `0` for local/offline runs. |
| `--resume` | `no` | Resume mode for Weights & Biases. Use `must` with `--run_id` to resume. |
| `--run_id` | empty | Weights & Biases run id for resumption. |
| `--entity` | `uaena` | Weights & Biases entity. |
| `--checkpoint_path` | `None` | Checkpoint path for resuming training. |

Additional prediction-only arguments in `main_pred.py`:

| Argument | Default | Description |
| --- | ---: | --- |
| `--model_path` | `None` | Required checkpoint path for test-time prediction. |
| `--path` | `../your/path/here` | Legacy checkpoint-path placeholder. |
| `--svgd_kernel` | `None` | Legacy SVGD kernel argument retained for compatibility. |
| `--potential_strength` | `5.0` | Orbital potential strength used by level-wise prediction. |

## Reproducibility Notes

Use the same tuple of `dataset`, `expID`, `negsamples`, and `seed` for training and prediction so both scripts load the same processed pickle. The processed file name is:

```text
taxonomy_data_<expID><negsamples><seed>_.pkl
```

For deterministic comparisons, keep `--seed`, dataset-specific overrides, pretrained backbone paths, and checkpoint paths fixed across runs.

## Questions
If you have any questions in running the code, you may email us and we will try to resolve it asap. 