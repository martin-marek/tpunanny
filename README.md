# TPU nanny

TPU nanny is designed to babysit spot TPUs. It will keep a number of spot TPUs running a user-specified SSH script, recreating any TPUs when they get preempted.

This library has similar goals to [tpucare](https://github.com/ClashLuke/tpucare/tree/main) and [tpunicorn](https://github.com/shawwn/tpunicorn); the main differences are that this library (1) uses queued resources to spin up TPUs; (2) is written in pure Python; and (3) has fewer features (hopefully making the source code more readable!).

# Minimal example

```python
import tpunanny as tn

tn.babysit(
    idxs=slice(1), # using a single TPU, index 0
    tpu_type='v6e-8',
    zone='europe-west4-a',
    project_id='my_gcs_project_id',
    ssh_script='echo "hello world"',
    startup_script='#!/bin/bash\necho "startup script ran" >> /tmp/startup.log',
)
```

`ssh_script` runs over SSH after the TPU becomes active. `startup_script` is passed as TPU metadata (`startup-script`) and runs when the TPU VM boots.

# Monitoring TPUs

I included a helper utility to monitor the state of all TPUs within a single Google Cloud project:

```bash
python monitor.py 'picodo-455019'
```

<img src="https://github.com/martin-marek/tpunanny/blob/main/figures/monitor.jpg" width="650">

# Setup

Install `requirements.txt` and ensure `gcloud` CLI is installed and authenticated:
```bash
uv pip install -r requirements.txt
gcloud auth login
```
