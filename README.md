# One Mode Rules Them All: Weighted Imitation Learning on a Single Mode

This is the repository for the paper *One Mode Rules Them All: Weighted Imitation Learning on a Single Mode*.

**Abstract**: Offline reinforcement learning (RL) aims to learn optimal policies from static datasets without active environment interaction. A key challenge in offline RL is that historical datasets often contain data from multiple sources, resulting in multi-modal action distributions conditioned on each state. Existing methods frequently assume unimodal behavior policies, leading to suboptimal performance when faced with inherently multi-modal data. Weighted imitation learning is a promising approach for offline RL that assigns different importance weights to state-action pairs when imitating the behavior policy. However, current weighted imitation methods struggle with multi-modal action distributions, often converging to averaged actions that may be suboptimal or even invalid. We propose a novel method called Weighted Imitation Learning on One Mode (LOM) to address this challenge. LOM first identifies and decomposes the different action modes using a Gaussian mixture model. It then selects the most promising mode via a novel hyper-policy based on expected returns. Finally, it performs weighted imitation learning specifically on the actions sampled from the selected mode. Theoretically, we show that LOM is derived from maximizing returns under KL-divergence constraints between the learned policy and the targeted action mode. Empirically, LOM outperforms existing offline RL methods on standard D4RL benchmarks, demonstrating the effectiveness of mode-focused learning in complex, multi-modal scenarios.


## Requirements

* python = 3.9.0
* d4rl = 1.1
* mujoco-py = 2.1.2.14
* pytorch = 2.3.0+cu121

## Usage

Run the following code to train LOM agent on the HalfCheetah-full-replay-v2:
```
$ python train.py --env_name HalfCheetah-full-replay-v2
```
To track the running logs with wandb, please use
```
$ python train.py --env_name HalfCheetah-full-replay-v2 --enable_wandb 1
```
