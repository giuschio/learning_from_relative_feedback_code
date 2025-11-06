# CueLearner: Bootstrapping and Local Policy Adaptation from Relative Feedback

Official code release accompanying the paper **CueLearner: Bootstrapping and Local Policy Adaptation from Relative Feedback**.  

Project page: https://giuschio.github.io/learning_from_relative_feedback/  
arXiv: https://arxiv.org/abs/2507.04730

## Repository Layout
- `tools/rl/`: runnable training scripts used in the paper (see descriptions inside each script).
- `configs_exp/`: configurations for the main experiments reported in the paper (billiards task only).
- `assets/checkpoints/`: reference checkpoints used as an expert annotator.

## Installation
Prerequisites: tested with python 3.9 on Ubuntu 22.04. Should run on any python >= 3.9.

Setup:
```bash
git clone https://github.com/giuschio/cue-learner-public.git
cd cue-learner-public
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

You can verify the setup with:
```bash
python -c "import cuesim, cuelearner; print(cuesim.__version__)"
```

## Running Experiments
Every script in `tools/rl/` consumes an OmegaConf YAML file describing the environment, policy, trainer hyperparameters, and paths to checkpoints. The general calling pattern is:
```bash
python tools/rl/<script>.py --config path/to/config.yaml --seed <seed>
```
All outputs (configs, checkpoints, TensorBoard logs) are stored under `experiments/` as specified by `trainer.log_dir`.

### Baseline RL
Train a DDQN agent directly in the three-ball billiards task:
```bash
python tools/rl/train.py --config configs/ddqn_baseline.yaml --seed 0
```
Set `config.policy_checkpoint` to resume from a saved model.

### Experiment 1: guiding exploration
#### Learning an exploration policy
Learn auxiliary policies that guide exploration using different feedback modalities:
- **Demonstrations/Corrections**:
  ```bash
  python tools/rl/train_dagger_exploration.py \
      --config configs_exp/2025_02_20_dagger_exploration_policies/dagger_policy.yaml \
      --seed 0
  ```
- **Scalar feedback**:
  ```bash
  python tools/rl/train_tamer_exploration.py \
      --config configs_exp/2025_02_24_tamer_exploration_policies/tamer_policy.yaml \
      --seed 0
  ```
- **Relative advice**:
  ```bash
  python tools/rl/train_improvement_advice.py \
      --config configs_exp/2024_09_25_FINAL_ADVICE_POLICIES/advice_policy.yaml \
      --seed 0
  ```

#### Training with the learned exploration policy
Combine a learned exploration policy with on-policy RL to bootstrap learning:
```bash
python tools/rl/train_w_advice.py \
    --config configs_exp/2024_09_26_FINAL_EXPLORATION/1K_labels/1K_labels_seed_0.yaml \
    --seed 0
```
Other configs (to use other types of exploration policies) can be found in `configs_exp/2024_09_26_FINAL_EXPLORATION`.


### Experiment 2: post-hoc adaptation
Use feedback to adapt a previously-learned policy to changes in the environment.
- **Residual learning from corrections**:
  ```bash
  python tools/rl/train_improvement_residual.py --seed 0
  ```
- **RL finetuning**:
  ```bash
  python tools/rl/train.py --seed 0
  ```
- **Relative advice**:
  ```bash
  python tools/rl/train_improvement_advice.py --seed 0
  ```
Configs are available in `experiments/2024_10_11_FINETUNING`

## Checkpoints
Every experiment reported in the paper can be trained end-to-end with the configs provided here. If you would like pre-trained checkpoints or other intermediate artifacts for a specific setup, request access through this [Google Form](https://forms.gle/pN8Ds8rr5CV83YLv8)—we typically respond within a couple of days.

## Citing
If you use our code in your research, please cite our paper as:

    @misc{schiavi2025cuelearnerbootstrappinglocalpolicy,
      title={CueLearner: Bootstrapping and local policy adaptation from relative feedback},
      author={Giulio Schiavi and Andrei Cramariuc and Lionel Ott and Roland Siegwart},
      year={2025},
      eprint={2507.04730},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.04730}
    }


## License
This project is released under the [MIT License](LICENSE).
