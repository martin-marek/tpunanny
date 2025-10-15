# TPU nanny

TPU nanny is designed to babysit spot TPUs. It will keep a number of spot TPUs running a user-specified bash script, recreating any TPUs when they get preempted.

This library has similar goals to [tpucare](https://github.com/ClashLuke/tpucare/tree/main) and [tpunicorn](https://github.com/shawwn/tpunicorn); the main differences are that this library (1) uses queued resources to spin up TPUs; (2) is written in pure Python; and (3) has fewer features (hopefully making the source code more readable!).

# Minimal example

```python
import tpunanny as tn

tn.babysit(
    idxs=slice(1), # using a single TPU, index 0
    tpu_type='v6e-1',
    zone='europe-west4-a',
    project_id='my_gcs_project_id',
    script='echo "hello world"',
)
```

# Example: wandb sweep

```python
setup_script = """
# install libraries
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax orbax numpy tqdm hydra-core wandb

# add ~/.local/bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# download dataset
mkdir -p ~/datasets
gsutil -m cp -r gs://llm-optim-europe-west4/fineweb_edu_gpt2_2.5B ~/datasets

# download training codebase
git clone --depth=1 https://github.com/martin-marek/picodo.git

# start wandb agent
tmux new-session -d "
cd ~/REPO_DIR
wandb login WANDB_KEY
wandb agent SWEEP_URL
"
"""
import tpunanny as tn
import numpy as np

tn.babysit(
    idxs=np.s_[:8], # TPUs 0...7
    tpu_type='v6e-1',
    zone='europe-west4-a',
    project_id='personal-project-451418',
    script=setup_script,
)
```
