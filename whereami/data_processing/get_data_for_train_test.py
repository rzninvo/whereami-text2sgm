# Go through 3DSSG, ScanScribe, Human datasets, and aggregate it with the point cloud datasets from 3RScan
# and make a dataset with all the graph-pointcloud-text pairs, split it into 75% train, 25% test
# The ScanScribe and human dataset should be able to index into the 3DSSG/3RScan dataset

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch


def scanscribe_get_text_cells(ids, texts, pc_data):
    text_scanscribe_text2pos = []
    scene_ids_for_cells = []
    for scene_id in ids:
        for text in texts[scene_id]:
            text_scanscribe_text2pos.append(text)
            scene_ids_for_cells.append(scene_id)

    cells_scanscribe_text2pos = []
    for scene_id in scene_ids_for_cells:
        cells_scanscribe_text2pos.append(pc_data[scene_id])
    assert(len(text_scanscribe_text2pos) == len(cells_scanscribe_text2pos))
    return text_scanscribe_text2pos, cells_scanscribe_text2pos

def human_get_text_cells(texts, pc_data):
    text_human_text2pos = []
    scene_ids_for_cells_human = []
    for text in tqdm(texts):
        text_human_text2pos.append(text['description'])
        scene_ids_for_cells_human.append(text['scanId'].split('/')[0])

    cells_human_text2pos = []
    for scene_id in tqdm(scene_ids_for_cells_human):
        cells_human_text2pos.append(pc_data[scene_id])
    assert(len(text_human_text2pos) == len(cells_human_text2pos))
    return text_human_text2pos, cells_human_text2pos

def get_scanscribe_graphs(training_scene_ids, testing_scene_ids, scanscribe_graphs):
    scanscribe_graphs_training = {scene_id: scanscribe_graphs[scene_id] for scene_id in training_scene_ids}
    scanscribe_graphs_testing = {scene_id: scanscribe_graphs[scene_id] for scene_id in testing_scene_ids}
    return scanscribe_graphs_training, scanscribe_graphs_testing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create train/test splits for graph datasets')
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get('WHEREAMI_DATA_ROOT', './data'),
                        help='Root data directory')
    parser.add_argument('--scanscribe_text', type=str, required=True,
                        help='Path to scanscribe_cleaned.json')
    parser.add_argument('--human_text', type=str, required=True,
                        help='Path to human data_json_format.json')
    parser.add_argument('--pc_data', type=str, default=None,
                        help='Path to cells_dict.pth (point cloud data)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated splits')
    cli_args = parser.parse_args()

    random.seed(11)

    data_root = Path(cli_args.data_root)
    graphs_dir = data_root / 'processed_data'

    _3dssg_graphs = torch.load(graphs_dir / '3dssg' / '3dssg_graphs_processed_edgelists_relationembed.pt')
    scanscribe_graphs = torch.load(graphs_dir / 'scanscribe' / 'scanscribe_cleaned_original_node_edge_features.pt')
    human_graphs = torch.load(graphs_dir / 'human' / 'human_graphs_processed.pt')

    with open(cli_args.scanscribe_text, 'r') as f:
        scanscribe_text = json.load(f)
    with open(cli_args.human_text, 'r') as f:
        human_text = json.load(f)

    training_ids_path = graphs_dir / 'training' / 'training_scene_ids.txt'
    testing_ids_path = graphs_dir / 'testing' / 'testing_scene_ids.txt'

    training_scene_ids = training_ids_path.read_text().splitlines()
    testing_scene_ids = testing_ids_path.read_text().splitlines()
    assert(len(set(training_scene_ids)) == len(training_scene_ids))
    assert(len(set(testing_scene_ids)) == len(testing_scene_ids))

    print(f'Training scenes: {len(training_scene_ids)}')
    print(f'Testing scenes: {len(testing_scene_ids)}')

    if cli_args.pc_data:
        _3dssg_pc = torch.load(cli_args.pc_data)
        testing_text_human, testing_cells_human = human_get_text_cells(human_text, _3dssg_pc)
        output_dir = Path(cli_args.output_dir) if cli_args.output_dir else data_root
        torch.save(testing_text_human, output_dir / 'testing_text_human_text2pos.pt')
        torch.save(testing_cells_human, output_dir / 'testing_cells_human_text2pos.pt')
