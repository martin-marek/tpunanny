import time
import subprocess
import threading
from google.cloud import tpu_v2
from google.api_core.exceptions import NotFound
client = tpu_v2.TpuClient()
_stop_event = threading.Event()


def get_runtime(tpu_type):
    # https://cloud.google.com/tpu/docs/runtimes
    if 'v6e' in tpu_type: return 'v2-alpha-tpuv6e'
    elif 'v5p' in tpu_type: return 'v2-alpha-tpuv5'
    elif 'v5lite' in tpu_type: return 'v2-alpha-tpuv5-lite'
    else: return 'tpu-ubuntu2204-base'


def _create(tpu_id, tpu_type, zone, project_id, startup_script=None):
    parent = f'projects/{project_id}/locations/{zone}'

    node_metadata = {}
    if startup_script is not None:
        node_metadata['startup-script'] = startup_script

    queued_resource = tpu_v2.QueuedResource(
        tpu=tpu_v2.QueuedResource.Tpu(
            node_spec=[
                tpu_v2.QueuedResource.Tpu.NodeSpec(
                    parent=parent,
                    node_id=tpu_id,
                    node=tpu_v2.Node(
                        accelerator_type=tpu_type,
                        runtime_version=get_runtime(tpu_type),
                        network_config=tpu_v2.NetworkConfig(enable_external_ips=True),
                        metadata=node_metadata,
                    ),
                )
            ]
        ),
        spot=tpu_v2.QueuedResource.Spot(),
    )

    operation = client.create_queued_resource(
        parent=parent,
        queued_resource_id=tpu_id,
        queued_resource=queued_resource,
    )
    
    return operation.result()


def _delete(tpu_id, zone, project_id):
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    request = tpu_v2.DeleteQueuedResourceRequest(name=qr_name, force=True)
    operation = client.delete_queued_resource(request=request)
    return operation


def _delete_all_suspended(project_id):
    """
    Deletes all queued resources in SUSPENDED state across all zones.
    Returns a list of dicts with 'tpu_id' and 'zone' for each deleted resource.
    """

    deleted, pending = [], []
    queued_resources = client.list_queued_resources(parent=f'projects/{project_id}/locations/-')
    for qr in queued_resources:
        zone = qr.name.split('/')[3]
        qr_id = qr.name.split('/')[-1]
        state = qr.state.state.name
        if state in ('FAILED', 'SUSPENDED'):
            pending.append((_delete(qr_id, zone, project_id), qr_id, zone))

    for operation, qr_id, zone in pending:
        try:
            operation.result()
            deleted.append({'tpu_id': qr_id, 'zone': zone})
            print(f'Deleted suspended queued resource {qr_id} in {zone}')
        except NotFound:
            print(f'Warning: queued resource {qr_id} in {zone} not found (already deleted?)')
        except Exception as e:
            print(f'Error: failed to delete {qr_id} in {zone}: {e}')

    return deleted


def _recreate(tpu_id, tpu_type, zone, project_id, startup_script=None):
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    
    try:
        # get TPU status
        tpu_info = client.get_queued_resource(name=qr_name)
        tpu_state = tpu_info.state.state.name
        
        # if TPU is unhealthy, delete it and create a new one
        if tpu_state in ('FAILED', 'SUSPENDED'):
            _delete(tpu_id, zone, project_id).result()
            time.sleep(30)
            _create(tpu_id, tpu_type, zone, project_id, startup_script)
            return 're-created'

    except NotFound as e:
        # if TPU doesn't exist, create it
        _create(tpu_id, tpu_type, zone, project_id, startup_script)
        return 'created'

    return 'exists'


def _run(tpu_id, zone, project_id, ssh_script):
    """Runs `ssh_script` on all workers of a TPU VM via gcloud SSH."""
    import os
    output_dir = f'logs/{zone}/{tpu_id}'
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_id,
        f'--zone={zone}',
        f'--project={project_id}',
        '--worker=all',
        f'--command={ssh_script}',
        f'--output-directory={output_dir}',
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _babysit(tpu_id, tpu_type, zone, project_id, stop_event, ssh_script=None, startup_script=None):
    """(Re)creates TPU and runs `ssh_script`."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    ran_ssh_script = False

    print(f'[{tpu_id}] starting to babysit...')
    while not stop_event.is_set():

        # check if TPU is healthy
        print(f'[{tpu_id}] checking TPU status...')
        create_status = _recreate(tpu_id, tpu_type, zone, project_id, startup_script)
        print(f'[{tpu_id}] TPU status: {create_status}')
        if create_status != 'exists': ran_ssh_script = False

        # if an SSH script was provided, wait until TPU is ready, then run it
        if not ran_ssh_script and ssh_script is not None:

            # wait until TPU is ready
            while not stop_event.is_set():
                tpu_info = client.get_queued_resource(name=qr_name)
                tpu_state = tpu_info.state.state.name
                print(f'[{tpu_id}] TPU state={tpu_state}')
                if tpu_state == 'ACTIVE': break
                stop_event.wait(10)

            if stop_event.is_set(): break

            # run ssh script
            print(f'[{tpu_id}] running ssh script...')
            result = _run(tpu_id, zone, project_id, ssh_script)
            print(f'[{tpu_id}] ssh script finished with exit code {result.returncode}.')
            ran_ssh_script = True

        # wait before checking on the TPU again
        print(f'[{tpu_id}] sleeping...')
        stop_event.wait(60)


def babysit(idxs, tpu_type, zone, project_id, ssh_script=None, startup_script=None):
    """Keeps multiple TPUs alive, optionally running `ssh_script` and `startup_script` on boot."""
    global _stop_event

    # stop any previously running babysit threads
    _stop_event.set()
    time.sleep(2)
    _stop_event = threading.Event()

    # create and start a thread for each TPU
    threads = []
    for idx in idxs:
        tpu_id = f'tn-{tpu_type}-{idx}'
        thread = threading.Thread(
            target=_babysit,
            args=(tpu_id, tpu_type, zone, project_id, _stop_event, ssh_script, startup_script),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        time.sleep(1) # stagger creation

    # keep main thread alive while threads are running
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    finally:
        _stop_event.set()
