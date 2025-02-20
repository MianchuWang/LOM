# Learning on One Mode: Addressing Multi-Modality in Offline Reinforcement Learning

This is the official code repository for the paper *Learning on One Mode: Addressing Multi-Modality in Offline Reinforcement Learning*.

**Abstract**: Offline reinforcement learning (RL) seeks to learn optimal policies from static datasets without interacting with the environment. A common challenge is handling multi-modal action distributions, where multiple behaviours are represented in the data. Existing methods often assume unimodal behaviour policies, leading to suboptimal performance when this assumption is violated. We propose Weighted Imitation Learning on One Mode (LOM), a novel approach that focuses on learning from a single, promising mode of the behaviour policy. By using a Gaussian mixture model to identify modes and selecting the best mode based on expected returns, LOM avoids the pitfalls of averaging over conflicting actions. Theoretically, we show that LOM improves performance while maintaining simplicity in policy learning. Empirically, LOM outperforms existing methods on standard D4RL benchmarks and demonstrates its effectiveness in complex, multi-modal scenarios.


## Requirements

* python = 3.9.0
* d4rl = 1.1
* mujoco-py = 2.1.2.14
* pytorch = 2.3.0+cu121

## Usage

Run the following code to train the LOM agent on HalfCheetah-full-replay-v2:
```
$ python train.py --env_name HalfCheetah-full-replay-v2
```
To track the running logs with wandb, please use
```
$ python train.py --env_name HalfCheetah-full-replay-v2 --enable_wandb 1
```
To replicate the results, please use
```
$ bash run_baselines.sh
```

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
