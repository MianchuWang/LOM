# Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning

This is the official code repository for the paper *Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning*.

**Abstract**: Offline reinforcement learning (RL) seeks to learn optimal policies from static datasets without interacting with the environment. A common challenge is handling multi-modal action distributions, where multiple behaviours are represented in the data. Existing methods often assume unimodal behaviour policies, leading to suboptimal performance when this assumption is violated. We propose Weighted Imitation Learning on One Mode (LOM), a novel approach that focuses on learning from a single, promising mode of the behaviour policy. By using a Gaussian mixture model to identify modes and selecting the best mode based on expected returns, LOM avoids the pitfalls of averaging over conflicting actions. Theoretically, we show that LOM improves performance while maintaining simplicity in policy learning. Empirically, LOM outperforms existing methods on standard D4RL benchmarks and demonstrates its effectiveness in complex, multi-modal scenarios.


## Requirements

* python = 3.9.0
* d4rl = 1.1
* mujoco-py = 2.1.2.14
* pytorch = 2.3.0+cu121

The MuJoCo/D4RL stack is legacy software and is most reliable on Linux or
WSL2 with Python 3.9. `wandb` is optional and is only required when using
`--enable_wandb 1`.

## Usage

Run the following code to train the LOM agent on HalfCheetah-full-replay-v2:
```
$ python train.py --env_name halfcheetah-full-replay-v2
```
To track the running logs with wandb, please use
```
$ python train.py --env_name halfcheetah-full-replay-v2 --enable_wandb 1
```
To replicate the results, please use
```
$ bash run_baselines.sh
```

## Top-k GNN extension

This branch adds an experimental mode-selection extension while preserving the
original LOM algorithm as the default. The three comparable configurations are:

| `--mode_selection` | Selection rule |
| --- | --- |
| `top1` | Original LOM: choose the highest Hyper-Q component. |
| `topk_q` | Keep the Hyper-Q top-k and rerank them using Monte Carlo estimates from the learned Q-function. |
| `topk_gnn` | Keep the Hyper-Q top-k and rerank them with a learned graph network. |
| `topk_gnn_merge` | Keep the Hyper-Q top-k and sample from a GNN-weighted mixture of them. |

The existing `K` hyperparameter still means the number of Q-networks. The new
`--top_k` argument controls the number of Gaussian modes and defaults to 3.

### Architecture and objective

For every state, the GMM produces all Gaussian components and the existing mode
network supplies their Hyper-Q scores. The selected top-k components become a
fully connected graph. Each node contains its Gaussian mean, standard deviation,
mixing coefficient, and original Hyper-Q score. Euclidean distances between
component means are edge features. Two PyTorch-native message-passing layers
produce one refined score per node; no graph library or new dependency is used.

The GNN target is a detached Monte Carlo estimate of the learned Q-value for each
selected Gaussian. Targets are normalized within each state. Training combines
Smooth L1 regression with a pairwise ranking loss (weighted by
`--gnn_ranking_weight`). Gradients update only the GNN in this step: the GMM,
Q-network, and original mode network retain their existing, separate training
objectives. This target makes the extension experimentally testable but is not
independent evidence that it improves returns.

In `topk_gnn_merge`, the GNN scores are converted into normalized mixture weights
and combined with the selected GMM weights. Action generation samples a component
from this categorical distribution and then samples from that component's Gaussian.
The component means are not averaged, which preserves distinct action modes. The
`--gnn_temperature` and `--gnn_prior_weight` arguments control score sharpness and
the influence of the original GMM weights. Merge entropy, maximum merge weight,
and effective mode count are logged under `gnn/*`.

### Commands

The original baseline remains the default:

```bash
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection top1
```

Run the two extensions with:

```bash
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection topk_q --top_k 3
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection topk_gnn --top_k 3
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection topk_gnn_merge --top_k 3
```

GMM pretraining is still 200,000 steps by default. It can be reused or shortened
for smoke tests:

```bash
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection topk_gnn \
  --gmm_checkpoint gmm_halfcheetah-medium-replay-v2.pth
python train.py --env_name halfcheetah-medium-replay-v2 --mode_selection topk_gnn \
  --gmm_pretrain_steps 10 --training_steps 10 --eval_every 10 --save_gmm 0
```

Run unit and GNN sanity tests with:

```bash
python -m unittest discover -s tests -v
```

For the full three-seed comparison, run `bash run_topk_experiments.sh`. Override
the defaults with `ENV_NAME=...` and `SEEDS="..."`. Use identical datasets,
training steps, evaluation intervals, and seeds across all three configurations.
Record episodic return, normalized score, variance across seeds, wall-clock time,
and computational overhead. Per-update timings are logged under `time/*`; compare
`time/train_step_seconds` across configurations and use `time/gnn_step_seconds`
to isolate the graph update.

### Current limitations

No benchmark result is reported yet. The GNN target depends on the learned
Q-function and can inherit its estimation error. The top-k Q baseline adds
sampling cost during policy training, and the graph method adds a GNN update.
Results should therefore be treated as an ablation study, not a demonstrated
improvement, until controlled D4RL experiments are completed.

## Reference

If you find our research helpful, please cite our paper at ICLR 2025:
```
@inproceedings{
    wang2025lom,
    title={Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning},
    author={Mianchu Wang and Yue Jin and Giovanni Montana},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=upkxzurnLC}
}
```
