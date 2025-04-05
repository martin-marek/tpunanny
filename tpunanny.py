import os
import time
import logging
import threading
from fabric import Connection
from google.cloud import tpu_v2
from google.api_core.exceptions import NotFound


# create TPU client
client = tpu_v2.TpuClient()


def create_tpu(tpu_name, tpu_type):
    node = tpu_v2.Node(
        accelerator_type=tpu_type,
        runtime_version='tpu-ubuntu2204-base',
        scheduling_config=tpu_v2.SchedulingConfig(preemptible=True),
        network_config=tpu_v2.NetworkConfig(enable_external_ips=True)
    )
    parent, tpu_id = tpu_name.split('/nodes/')
    operation = client.create_node(
        parent=parent,
        node_id=tpu_id,
        node=node,
    )
    
    # block thread until created
    return operation.result()


def run_script(tpu_ip, script, hide=False):
    """Connects to TPU using SSH and runs `script`."""
    result = Connection(tpu_ip).run(script, shell='/bin/bash -l', hide=hide)
    return result


def _babysit_tpu(client, project_id, zone, tpu_id, tpu_type, setup_script):
    """(Re)creates TPU and runs `setup_script`."""
    tpu_name = f'projects/{project_id}/locations/{zone}/nodes/{tpu_id}'

    # create logger for this TPU
    logger = logging.getLogger(tpu_id)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(f'logs/{tpu_id}.txt', 'w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # run an infinite loop that maintains the TPU
    logger.info(f'Starting to babysit {tpu_name}...')
    while True:
    
        # check if tpu exists
        logger.info(f'checking if TPU exists...')
        try:
            tpu_info = client.get_node(name=tpu_name)
            logger.info(f'TPU found, state={tpu_info.state.name}, health={tpu_info.health.name}')

            tpu_exists = True
        except NotFound as e:
            logger.info(f'TPU not found')
            tpu_exists = False
    
        # create new tpu
        if not tpu_exists:
            logger.info('(re)creating TPU...')
            try:
                # create tpu
                create_tpu(tpu_name, tpu_type)
        
                # get ip address
                tpu_info = client.get_node(name=tpu_name)
                tpu_ip = tpu_info.network_endpoints[0].ip_address
            
                # run setup script
                logger.info('running script...')
                result = run_script(tpu_ip, setup_script, hide=True)
                # logger.info(f"script stdout: {result.stdout}")
                # logger.info(f"script stderr: {result.stderr}")
            
            except Exception as e:
                logger.error(e)

        # wait before checking on the TPU again
        time.sleep(300)


def babysit_tpus(prefix, num_tpus, tpu_type, zone, project_id, setup_script):
    """Keeps multiple TPUs alive and running `setup_script`."""

    # crete output directory for logs
    if not os.path.exists('logs'): os.mkdir('logs')

    # create and start a thread for each TPU
    threads = []
    for i in range(num_tpus):
        tpu_id = f'{prefix}-{i}'
        thread = threading.Thread(target=_babysit_tpu, args=(client, project_id, zone, tpu_id, tpu_type, setup_script), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(60) # stagger creation
    
    # keep main thread alive
    while any(t.is_alive() for t in threads):
        time.sleep(60)
