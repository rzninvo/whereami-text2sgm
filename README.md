# Where am I?: Scene Retrieval with Language

This is the repository for [Where am I?: Scene Retrieval with Language](https://whereami-langloc.github.io/).

Given a natural-language description of a scene (e.g. *"I'm standing in front of a desk with a monitor, next to a bookshelf"*), the system:

1. **Retrieves** the matching 3D scene from a database of 3DSSG scene graphs.
2. **Localises** where the person was standing (and looking) inside that scene.

---

## Setup

### 1. Create a conda environment

```bash
conda create -n whereami python=3.10 -y
conda activate whereami
```

### 2. Install PyTorch with CUDA

Pick the command matching your CUDA driver from [pytorch.org](https://pytorch.org/get-started/locally/). For example:

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric
```

### 4. Install the package

```bash
pip install -e .
```

This installs all remaining dependencies (spaCy, Open3D, W&B, Hydra, OpenAI, etc.) and makes `whereami` importable from anywhere.

### 5. Download the spaCy language model

```bash
python -m spacy download en_core_web_lg
```

### 6. Set up your API key (optional)

An OpenAI API key is only needed for free-text query inference (`single_inference`) and the Ada embedding backend. If you only plan to train/evaluate with the default word2vec embeddings, you can skip this.

```bash
cp .env.example .env
# edit .env and paste your key:
#   OPENAI_API_KEY=sk-...
```

The key is loaded automatically at import time via `python-dotenv`.

---

## Data

Download the [processed graph data](https://drive.google.com/drive/folders/1rg-MTfvAxT7s_mukAioYnPpzpbQrI7Ji?usp=sharing) and place everything under `data/`:

```text
data/
  processed_data/
    3dssg/              -- 3DSSG scene graphs
    training/           -- ScanScribe training split
    testing/            -- ScanScribe test split
    scanscribe/         -- ScanScribe text graphs
    human/              -- human-authored caption graphs
    sgfusion/           -- SGFusion (ScanNet) graphs
  model_checkpoints/
    graph2graph/        -- trained BigGNN checkpoints
```

Download the [pre-trained model weights](https://drive.google.com/file/d/1Ol1LuPVIVXvSXPmuMoEc5sIg20fCJ_su/view?usp=sharing) and place them in `data/model_checkpoints/graph2graph/`.

For visualization and localization you also need the 3RScan meshes:

```text
data/
  3rscan/
    <scene-id>/         -- mesh + instance segmentation files
```

---

## Usage

Everything is configured through [Hydra](https://hydra.cc/). Default values live in `conf/` and can be overridden from the command line with `key=value` syntax. Run any command with `--help` to see all available options.

### Training

```bash
python -m whereami.models.train \
    train.model_name=my_model \
    train.epoch=100 \
    train.batch_size=16 \
    train.entire_training_set=true
```

Key options:

| Flag | Default | Description |
| --- | --- | --- |
| `train.model_name` | *required* | Name for saved checkpoints |
| `train.epoch` | 30 | Number of training epochs |
| `train.lr` | 0.0001 | Learning rate |
| `train.batch_size` | 16 | Pairs per contrastive batch |
| `train.folds` | 10 | Cross-validation folds |
| `train.entire_training_set` | false | Skip CV, train on everything |
| `train.subgraph_ablation` | false | Disable DBSCAN subgraph matching |
| `mode` | online | W&B logging: online / offline / disabled |

### Evaluation

```bash
python -m whereami.models.eval \
    eval.model_name=my_model \
    eval.eval_iters=100
```

Reports Top-K retrieval accuracy (default K = 1, 2, 3, 5) averaged over multiple random draws.

### Inference

**Batch** -- score all ScanScribe captions against the 3DSSG database:

```bash
python -m whereami.models.inference \
    eval.model_name=my_model \
    inference.top_k=5
```

**Single query** -- type a free-text description and get the top matching scenes:

```bash
python -m whereami.models.single_inference \
    eval.model_name=my_model \
    inference.query="There is a wooden desk with a monitor next to a bookshelf" \
    inference.top_k=5
```

This calls the OpenAI API to parse your text into a scene graph, so make sure your `.env` is set up.

### Localization

Given matched objects in a scene, estimate the camera pose on a dense grid:

```bash
# Standard grid scoring
python -m whereami.localization.cli \
    localization.mode=standard

# Coarse-to-fine refinement
python -m whereami.localization.cli \
    localization.mode=coarse_to_fine

# Export ranked candidates as JSON
python -m whereami.localization.cli \
    localization.mode=candidates \
    localization.output_json=candidates.json
```

### Visualization

**Matched objects** -- highlight which scene objects match a caption:

```bash
python -m whereami.visualization.visualization_minimal \
    scan_id=<scene-id> \
    eval.model_name=my_model
```

**Full scoring** -- show best/worst/ground-truth scene matches for each caption:

```bash
python -m whereami.visualization.visualization_graph_object \
    scan_id=<scene-id> \
    eval.model_name=my_model
```

**Localization heatmap from free text** -- parse a query and visualise the predicted pose distribution:

```bash
python -m whereami.visualization.visualize_loc_from_query \
    scan_id=<scene-id> \
    inference.query="I can see a sofa facing a TV and a coffee table between them." \
    localization.show_heatmap=true \
    localization.show_3d=true
```

---

## Configuration

All settings live in YAML files under `conf/`. Every parameter is documented inline.

| Config group | File | What it controls |
| --- | --- | --- |
| Top-level | `conf/config.yaml` | W&B mode, seed, device, scan_id |
| Paths | `conf/paths/default.yaml` | Data directories and graph file paths |
| Model | `conf/model/default.yaml` | BigGNN architecture (layers, heads, dropout) |
| Graph | `conf/graph/default.yaml` | Embedding type, max distance, DBSCAN params |
| Training | `conf/train/default.yaml` | Epochs, LR, batch size, ablation flags |
| Evaluation | `conf/eval/default.yaml` | Top-K settings, iteration counts |
| Inference | `conf/inference/default.yaml` | Score blending, query text, debug mode |
| Localization | `conf/localization/default.yaml` | Grid resolution, FOV, prediction strategy |

Override anything from the CLI:

```bash
python -m whereami.models.train \
    model.N=2 model.heads=4 graph.embedding_type=clip train.epoch=50
```

---

## Scripts

Pre-configured shell scripts live in `scripts/` for common workflows:

| Script | What it does |
| --- | --- |
| `scripts/run.sh` | Train with subgraph ablation on the full dataset |
| `scripts/run_eval.sh` | Evaluate a trained checkpoint |
| `scripts/run_inference_scanscribe.sh` | Batch inference on ScanScribe captions |
| `scripts/visualize.sh` | Visualise matched objects for a single scene |
| `scripts/visualize_loc_prob.sh` | Free-text query localisation with heatmap |
| `scripts/visualize_eval_loc.sh` | Run localisation evaluation with metrics output |

Edit the `model_name` and `scan_id` values inside each script to match your setup, then run:

```bash
bash scripts/run.sh
```

---

## Baselines

- **CLIP2CLIP**: see `baselines/CLIP-to-CLIP/`
- **Text2Pos**: see this [fork](https://github.com/jiaqchen/Text2Pos-CVPR2022)
  - Fine-tuned weights: [download](https://drive.google.com/file/d/1Bkev47FdHgiLFF2-4BOMhp4P8W0ZOnNh/view?usp=sharing)
  - Trained from scratch on 3DSSG: [download](https://drive.google.com/file/d/1gJUF9Tgdket1ebu8gQsJyrd59MN3VJI8/view?usp=sharing)
