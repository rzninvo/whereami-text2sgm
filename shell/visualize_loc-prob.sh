#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
# PROJECT_DIR="/cluster/project/cvg/jiaqchen/h_coarse_loc/playground/graph_models/models/"
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"


# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# python3 visualization_minimal.py \
#     --scan-id 095821f7-e2c2-2de1-9568-b9ce59920e29 \
#     --root /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
#     --graphs /home/klrshak/Downloads/3RScan-Test/graphs/ \
#     --ckpt  '/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/model_checkpoints/graph2graph/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt'

# python visualize_loc_prob.py \
#   --root /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
#   --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data \
#   --top_k 5 --grid_step 0.25 \
#   --show_heatmap \
#   --show_arrows --h_fov_deg 100 --v_fov_deg 60 --arrow_stride 1

python visualize_loc_from_query.py \
  --root /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
  --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data \
  --scan_id c7895f0b-339c-2d13-80e2-1d2aa04aa528 \
  --query "This is probably a wardrobe or storage room. I see many cabinets. There is a shelf attached to the wall holding a number of boxes. I also see cleaning tools, like a mop and ironing board, hanging from the wall. There are a few greenish bags on the floor, reasonably small ones. " \
  --top_k 8 --grid_step 0.25 --show_heatmap --show_arrows --show_3d \
  --h_fov_deg 100 --v_fov_deg 60 \
  --api_key_file /home/klrshak/openai_api_key.txt

