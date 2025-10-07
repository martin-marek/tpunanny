import os
import sys
import time
import logging
import threading
from fabric import Connection
from google.cloud import tpu_v2
from google.api_core.exceptions import NotFound
client = tpu_v2.TpuClient()


def get_runtime(tpu_type):
    # https://cloud.google.com/tpu/docs/runtimes
    if 'v6e' in tpu_type: return 'v2-alpha-tpuv6e'
    elif 'v5p' in tpu_type: return 'v2-alpha-tpuv5'
    elif 'v5lite' in tpu_type: return 'v2-alpha-tpuv5-lite'
    else: return 'tpu-ubuntu2204-base'


def _create(tpu_type, tpu_id, zone, project_id):
    parent = f'projects/{project_id}/locations/{zone}'

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
    operation = client.delete_queued_resource(name=qr_name)
    return operation.result()


def _run(tpu_id, zone, project_id, script, hide=False):
    """Connects to TPU using SSH and runs `script`."""
    tpu_name = f'projects/{project_id}/locations/{zone}/nodes/{tpu_id}'
    tpu_info = client.get_node(name=tpu_name)
    # internal_ip_address = tpu_info.network_endpoints[0].ip_address
    external_ip_address = tpu_info.network_endpoints[0].access_config.external_ip
    result = Connection(external_ip_address).run(script, shell='/bin/bash -l', hide=hide)
    return result


def _babysit(tpu_type, tpu_id, zone, project_id, script=None, stream_log=True):
    """(Re)creates TPU and runs `script`."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'

    # create logger for this TPU
    logger = logging.getLogger(tpu_id)
    logger.setLevel(logging.DEBUG)

    # remove previous log handlers
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    
    # add file handler
    log_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(f'logs/{tpu_id}.txt', 'w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # optionally add stdout handler
    if stream_log:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_format)
        logger.addHandler(stream_handler)

    # run an infinite loop that maintains the TPU
    logger.info(f'starting to babysit {tpu_id}...')
    while True:
        
        logger.info('checking TPU status...')
        try:
            # try to get TPU status
            tpu_info = client.get_queued_resource(name=qr_name)
            tpu_state = tpu_info.state.state.name
            logger.info(f'TPU found, state={tpu_state}')

            # if TPU was preempted, delete it
            if tpu_state in ('FAILED', 'SUSPENDED'):
                logger.info('deleting TPU...')
                _delete(tpu_id, zone, project_id)
        
        except NotFound as e:
            # if we failed to get TPU status, the TPU doesn't exist -> create it
            logger.info('TPU not found')
            logger.info('creating TPU...')

            # create tpu
            _create(tpu_type, tpu_id, zone, project_id)

            # if script was provided, wait until TPU is ready, then run the script
            if script is not None:
            
                # wait until TPU is ready
                while True:
                    tpu_info = client.get_queued_resource(name=qr_name)
                    tpu_state = tpu_info.state.state.name
                    logger.info(f'TPU state={tpu_state}')
                    if tpu_state == 'ACTIVE': break
                    time.sleep(10)
                
                # run script
                logger.info('running script...')
                result = _run(tpu_id, zone, project_id, script, hide=True)
                logger.info(f"script stdout: {result.stdout}")
                logger.info(f"script stderr: {result.stderr}")

        # wait before checking on the TPU again
        logger.info('sleeping...')
        time.sleep(60)


def babysit(num_tpus, tpu_type, zone, project_id, script=None):
    """Keeps multiple TPUs alive and running `script`."""

    # crete output directory for logs
    if not os.path.exists('logs'): os.mkdir('logs')

    # create and start a thread for each TPU
    threads = []
    for i in range(num_tpus):
        tpu_id = f'tn-{tpu_type}-{idx}'
        thread = threading.Thread(target=_babysit, args=(tpu_type, tpu_id, zone, project_id, script, False), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(5) # stagger creation
    
    # keep main thread alive while threads are running
    while any(t.is_alive() for t in threads):
        time.sleep(1)
