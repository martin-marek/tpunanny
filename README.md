# TPU nanny

TPU nanny is designed to babysit spot TPUs. It will keep a number of spot TPUs running a user-specified bash script, recreating any TPUs when they get preempted.

This library has similar goals to [tpucare](https://github.com/ClashLuke/tpucare/tree/main) and [tpunicorn](https://github.com/shawwn/tpunicorn); the main differences are that this library (1) uses queued resources to spin up TPUs; (2) is written in pure Python; and (3) has fewer features (hopefully making the source code more readable!).

# Minimal example

```python
import tpunanny as tn

tn.babysit(
    idxs=slice(1), # using a single TPU, index 0
    tpu_type='v6e-8',
    zone='europe-west4-a',
    project_id='my_gcs_project_id',
    script='echo "hello world"',
)
```

# Setup

First, install `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

Second, add your SSH public key to Google Cloud. [To do this](https://github.com/ayaka14732/tpu-starter?tab=readme-ov-file#42-add-an-ssh-public-key-to-google-cloud), type “SSH keys” into the Google Cloud webpage search box, go to the relevant page, then click edit, and add your computer's SSH public key.
