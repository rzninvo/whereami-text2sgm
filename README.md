# Where am I?: Scene Retrieval with Language

This is the repository that contains source code for the work [Where am I?: Scene Retrieval with Language](https://whereami-langloc.github.io/).

Given a natural language description of a scene (e.g. *"I'm standing in front of a desk with a monitor, next to a bookshelf"*), the system:
1. **Retrieves** the matching 3D scene from a database of 3DSSG scene graphs.
2. **Localises** where the person was standing (and looking) inside that scene.

## Repository Structure

```
whereami-text2sgm/
  pyproject.toml                    # Package metadata & dependencies
  conf/                             # Hydra config files
    config.yaml                     # Top-level defaults
    paths/default.yaml              # Data & checkpoint paths
    train/default.yaml              # Training hyperparameters
    eval/default.yaml               # Evaluation parameters
    inference/default.yaml          # Inference parameters
    localization/default.yaml       # Localization parameters
    embedding/default.yaml          # Embedding config
  whereami/                         # Main Python package
    data_processing/                # Data loading & graph construction
      scene_graph.py                # Core SceneGraph, Node, Edge classes
      scene_graph_utils.py          # Graph validation helpers
      graph_loader_3dssg.py         # Build 3DSSG scene graphs from raw JSON
      graph_loader_scanscribe.py    # Build ScanScribe text graphs
      graph_loader_human.py         # Build human annotation graphs
      graph_loader_utils.py         # Shared embedding & geometry helpers
      create_text_embeddings.py     # Embedding backends (word2vec, CLIP, Ada)
      get_data_for_train_test.py    # Train/test split generation
    models/                         # Model, training, evaluation, inference
      model_graph2graph.py          # BigGNN: self-attn + cross-attn + MLP
      train.py                      # Training pipeline (contrastive loss)
      train_utils.py                # K-fold split, custom cross-entropy
      eval.py                       # Retrieval evaluation (Top-K accuracy)
      inference.py                  # Batch text-graph to scene retrieval
      single_inference.py           # Free-text query via LLM parsing
      args.py                       # CLI argument definitions
      timing.py                     # Timer utility
    analysis/                       # Subgraph matching & overlap analysis
      helper.py                     # DBSCAN-based subgraph extraction
    localization/                   # Dense-grid camera localization
      __init__.py                   # Public API re-exports
      grid.py                       # Grid sampling, raycasting, visibility
      matching.py                   # Cosine-similarity Top-K matcher
      frame_io.py                   # Frame JSON loading & caption graph building
      prediction.py                 # Prediction strategies & candidate building
      metrics.py                    # SceneMetrics, Hit@r, mass-radius, IoU
      coarse_search.py              # Multi-level coarse-to-fine arrow search
      visualization.py              # Colour maps, FOV geometry, plot helpers
      evaluation.py                 # Unified evaluate_scene + run_evaluation
      cli.py                        # Hydra CLI entry point
    visualization/                  # Standalone visualization scripts
      visualize_loc_prob.py         # Dense grid localization demo (thin wrapper)
      visualize_loc_from_query.py   # Free-text query localization
      visualize_3rscan_segments.py  # Segmented mesh builder/viewer
      visualization_graph_object.py # Matched scene/object visualiser
      visualization_minimal.py      # Lightweight one-scene visualiser
    utils/                          # General utilities
      utils.py                      # Word vectors, cross-graph builder, masking
  tests/                            # Unit tests
    test_scene_graph.py             # Tests for SceneGraph class
    test_utils.py                   # Tests for utility functions
  scripts/                          # Shell entry scripts
    run.sh                          # Launch training
    run_eval.sh                     # Launch evaluation
    run_inference_scanscribe.sh     # Launch batch inference
    visualize.sh                    # Launch visualization
    visualize_eval_loc.sh           # Launch localization evaluation
    visualize_loc_prob.sh           # Launch free-text localization
  baselines/                        # Baseline comparisons
    CLIP-to-CLIP/                   # CLIP-based retrieval baseline
```

## Setup

### 1. Create a conda environment

```bash
conda create -n whereami python=3.10 -y
conda activate whereami
```

### 2. Install PyTorch (with CUDA)

Install the version matching your CUDA driver. For example, for CUDA 11.8:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Or for CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric
```

### 4. Install the package

```bash
cd whereami-text2sgm
pip install -e .
```

This installs all remaining dependencies (spacy, open3d, wandb, hydra, openai, transformers, etc.) and makes the `whereami` package importable from anywhere.

### 5. Download the spaCy language model

```bash
python -m spacy download en_core_web_lg
```

### 6. Set environment variables

The repo uses environment variables for data paths instead of hardcoded paths:

```bash
# Required: root directory containing processed_data/, model_checkpoints/, etc.
export WHEREAMI_DATA_ROOT=/path/to/your/data

# Required for localization scripts: 3RScan mesh data directory
export RSCAN_ROOT=/path/to/3RScan/data/3RScan

# Optional: OpenAI API key (for Ada embeddings or LLM-based text parsing)
export OPENAI_API_KEY=sk-...
```

### 7. Download data & model weights

Download the model weights from [here](https://drive.google.com/file/d/1Ol1LuPVIVXvSXPmuMoEc5sIg20fCJ_su/view?usp=sharing) and place them in `$WHEREAMI_DATA_ROOT/model_checkpoints/graph2graph/`.

Download the processed graph data from [here](https://drive.google.com/drive/folders/1rg-MTfvAxT7s_mukAioYnPpzpbQrI7Ji?usp=sharing) and place the contents into `$WHEREAMI_DATA_ROOT/`.

## Usage

### Training

```bash
bash scripts/run.sh
```

Or directly:

```bash
python -m whereami.models.train \
    --epoch 100 --N 1 --lr 0.0001 --batch_size 16 \
    --entire_training_set --model_name my_model
```

### Evaluation

```bash
bash scripts/run_eval.sh
```

### Inference

Batch inference on ScanScribe captions:

```bash
python -m whereami.models.inference \
    --graphs $WHEREAMI_DATA_ROOT/processed_data \
    --ckpt $WHEREAMI_DATA_ROOT/model_checkpoints/graph2graph/best_model.pt \
    --top_k 5 --device cuda
```

Free-text query:

```bash
python -m whereami.models.single_inference \
    --graphs $WHEREAMI_DATA_ROOT/processed_data \
    --ckpt $WHEREAMI_DATA_ROOT/model_checkpoints/graph2graph/best_model.pt \
    --query "I see a wooden desk with a monitor next to a bookshelf" \
    --top_k 5
```

### Localization

Run the unified localization evaluation (standard, coarse-to-fine, or candidates mode):

```bash
# Standard mode (default)
python -m whereami.localization.cli \
    localization.root=$RSCAN_ROOT \
    localization.graphs=$WHEREAMI_DATA_ROOT/processed_data

# Coarse-to-fine mode
python -m whereami.localization.cli \
    localization.mode=coarse_to_fine \
    localization.root=$RSCAN_ROOT \
    localization.graphs=$WHEREAMI_DATA_ROOT/processed_data

# Candidates export mode
python -m whereami.localization.cli \
    localization.mode=candidates \
    localization.root=$RSCAN_ROOT \
    localization.graphs=$WHEREAMI_DATA_ROOT/processed_data \
    localization.output_json=./eval/candidates.json
```

Or use the helper script:

```bash
bash scripts/visualize_eval_loc.sh
```

### Tests

```bash
pytest tests/ -v
```

## Baselines

The CLIP2CLIP baseline can be found in `baselines/CLIP-to-CLIP/`. The Text2Pos baseline can be found in this [fork](https://github.com/jiaqchen/Text2Pos-CVPR2022).

Model weights for the fine-tuned Text2Pos are [here](https://drive.google.com/file/d/1Bkev47FdHgiLFF2-4BOMhp4P8W0ZOnNh/view?usp=sharing), and for the version trained from scratch on 3DSSG is [here](https://drive.google.com/file/d/1gJUF9Tgdket1ebu8gQsJyrd59MN3VJI8/view?usp=sharing).
