"""Evaluation script for trained BigGNN models on ScanScribe, human, and ScanNet test sets."""

import time
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import random

from whereami.data_processing.scene_graph import SceneGraph
from whereami.models.model_graph2graph import BigGNN
from whereami.models.train import eval_acc as eval_fn
from whereami.models.train import format_to_latex
from whereami.models.timing import Timer

if __name__ == "__main__":
    from whereami.models.args import get_args
    args = get_args()

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.current_device())
    random.seed(42)

    data_root = Path(args.data_root)
    graphs_dir = data_root / 'processed_data'
    ckpt_dir = data_root / 'model_checkpoints' / 'graph2graph'
    eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else data_root / 'eval_outputs'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    wandb.config = { "architecture": "self attention cross attention",
                     "dataset": "ScanScribe_cleaned"}
    for arg in vars(args): wandb.config[arg] = getattr(args, arg)
    wandb.init(project="graph2graph",
                mode=args.mode,
                config=wandb.config)

    # 3DSSG
    _3dssg_graphs = {}
    _3dssg_scenes = torch.load(graphs_dir / '3dssg' / '3dssg_graphs_processed_edgelists_relationembed.pt')
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid,
                                            graph_type='3dssg',
                                            graph=_3dssg_scenes[sceneid],
                                            max_dist=1.0, embedding_type='word2vec',
                                            use_attributes=args.use_attributes)

    # ScanScribe Test
    scanscribe_graphs_test = {}
    scanscribe_scenes = torch.load(graphs_dir / 'scanscribe' / 'scanscribe_text_graphs_from_image_desc_node_edge_features.pt')
    for scene_id in tqdm(scanscribe_scenes):
        scanscribe_graphs_test[scene_id] = SceneGraph(scene_id,
                                                txt_id=None,
                                                graph_type='human',
                                                graph=scanscribe_scenes[scene_id],
                                                embedding_type='word2vec',
                                                use_attributes=args.use_attributes)

    print(f'number of scanscribe test graphs before removing: {len(scanscribe_graphs_test)}')
    to_remove = []
    for g in scanscribe_graphs_test:
        if len(scanscribe_graphs_test[g].edge_idx[0]) < 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs_test[g]
    print(f'number of scanscribe test graphs after removing: {len(scanscribe_graphs_test)}')

    # Human Test
    h_graphs_test = torch.load(graphs_dir / 'human' / 'human_graphs_processed.pt')
    h_graphs_remove = [k for k in h_graphs_test if k.split('_')[0] not in _3dssg_graphs]
    print(f'to remove human_graphs, hopefully none: {h_graphs_remove}')
    for k in h_graphs_remove: del h_graphs_test[k]
    assert all([k.split('_')[0] in _3dssg_graphs for k in h_graphs_test]), \
        "All human graph scene IDs must exist in 3DSSG"
    human_graphs_test = {k: SceneGraph(k.split('_')[0],
                                   graph_type='human',
                                   graph=h_graphs_test[k],
                                   embedding_type='word2vec',
                                   use_attributes=args.use_attributes) for k in h_graphs_test}

    scannet_test_graphs = torch.load(graphs_dir / 'sgfusion' / 'sgfusion_graphs.pt')
    scannet_test_graphs = {k: SceneGraph(k,
                                      graph_type='human',
                                      graph=scannet_test_graphs[k],
                                      embedding_type='word2vec',
                                      use_attributes=args.use_attributes) for k in scannet_test_graphs}

    scannet_test_text_graphs = torch.load(graphs_dir / 'sgfusion' / 'sgfusion_text_graphs.pt')
    scannet_test_text_graphs = {k: SceneGraph(k,
                                        graph_type='human',
                                        graph=scannet_test_text_graphs[k],
                                        embedding_type='word2vec',
                                        use_attributes=args.use_attributes) for k in scannet_test_text_graphs}

    model_name = args.model_name
    model_state_dict = torch.load(ckpt_dir / f'{model_name}.pt')
    model = BigGNN(args.N, args.heads).to('cuda')
    model.load_state_dict(model_state_dict)

    if args.eval_entire_dataset:
        model_name = model_name + '_topkoutofentire_'
    if args.eval_only_c:
        model_name = model_name + '_eval_only_c'
    if args.scannet:
        model_name = model_name + '_scannet_'
    if args.scanscribe_auto_gen:
        model_name = model_name + '_scanscribe_auto_gen_'
    model_name = model_name + '_' + str(args.eval_iters)

    start = time.time()
    scanscribe_timer = Timer()
    scanscribe_test_acc = eval_fn(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(scanscribe_graphs_test.values()),
                                    fold=None,
                                    args=args,
                                    mode='scanscribe_test',
                                    timer=scanscribe_timer)
    print(f'accuracy on scanscribe test set: {scanscribe_test_acc}')
    end_scanscribe = time.time()
    print(f'time for scanscribe test set: {end_scanscribe - start}')
    with open(eval_output_dir / f'{model_name}_scanscribe_test_acc.txt', 'w') as f:
        scanscribe_test_acc = format_to_latex(scanscribe_test_acc)
        f.write(f'{scanscribe_test_acc}')

    start = time.time()
    human_timer = Timer()
    human_test_acc = eval_fn(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(human_graphs_test.values()),
                                    fold=None,
                                    args=args,
                                    mode='human_test',
                                    valid_top_k=args.valid_top_k,
                                    timer=human_timer)
    print(f'accuracy on human test set: {human_test_acc}')
    end_human = time.time()
    print(f'time for human test set: {end_human - start}')
    with open(eval_output_dir / f'{model_name}_human_test_acc.txt', 'w') as f:
        human_test_acc = format_to_latex(human_test_acc)
        f.write(f'{human_test_acc}')
