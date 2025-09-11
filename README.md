# TPU nanny

**🚨DO NOT USE – THIS LIBRARY IS OUTDATED!🚨**
---

TPU nanny is designed to babysit preemptible TPUs. It will keep a number of preemptible TPUs running a user-specified bash script, recreating any TPUs when they get preempted.

This library has similar goals to [tpucare](https://github.com/ClashLuke/tpucare/tree/main) and [tpunicorn](https://github.com/shawwn/tpunicorn); the main differences are that this library is (1)written in pure Python; and (2) has fewer features (hopefully making the code more readable).

# Minimal example

```python
import tpunanny

tpunanny.babysit_tpus(
    prefix='tpunanny-test-1',
    num_tpus=1,
    tpu_type='v3-8',
    zone='europe-west4-a',
    project_id='my_gcs_project_id',
    setup_script='echo "hello world"',
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

# clone repo
git clone --depth=1 https://github.com/martin-marek/picodo.git

# start wandb agent
tmux new-session -d "
cd ~/REPO_DIR
wandb login WANDB_KEY
wandb agent SWEEP_URL
"
"""
import tpunanny

tpunanny.babysit_tpus(
    prefix='sweep-1',
    num_tpus=10,
    tpu_type='v3-8',
    zone='europe-west4-a',
    project_id='personal-project-451418',
    setup_script=setup_script
)
```
