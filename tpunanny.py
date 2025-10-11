import os
import sys
import time
import logging
import threading
from fabric import Connection
from concurrent.futures import ThreadPoolExecutor
from google.cloud import tpu_v2
from google.api_core.exceptions import NotFound
client = tpu_v2.TpuClient()


class LoggerWriter:
    """A file-like object that writes to a logger."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        # Log only non-empty messages to avoid spamming newlines
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        # This method is required for file-like object compatibility
        pass


def get_runtime(tpu_type):
    # https://cloud.google.com/tpu/docs/runtimes
    if 'v6e' in tpu_type: return 'v2-alpha-tpuv6e'
    elif 'v5p' in tpu_type: return 'v2-alpha-tpuv5'
    elif 'v5lite' in tpu_type: return 'v2-alpha-tpuv5-lite'
    else: return 'tpu-ubuntu2204-base'


def _create(tpu_id, tpu_type, zone, project_id):
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
    """
    This function can only delete TPU in the following states:
    (ACCEPTED, WAITING_FOR_RESOURCES, SUSPENDED, FAILED)
    """
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    operation = client.delete_queued_resource(name=qr_name)
    return operation.result()


def _run(tpu_id, zone, project_id, script, out_stream=None, err_stream=None):
    """
    Connects to TPU using SSH and runs `script`.
    Run script on all hosts, return output only from first host.
    """
    tpu_name = f'projects/{project_id}/locations/{zone}/nodes/{tpu_id}'
    tpu_info = client.get_node(name=tpu_name)
    ips = [endpoint.access_config.external_ip for endpoint in tpu_info.network_endpoints]
    run_on_host = lambda ip, out, err, hide: Connection(ip).run(
        script, shell='/bin/bash -l',
        out_stream=out,
        err_stream=err,
        hide=hide,
        warn=True,
    )
    with ThreadPoolExecutor() as executor:    
        futures = [
            executor.submit(
                run_on_host,
                ip,
                out_stream if i == 0 else None,
                err_stream if i == 0 else None,
                i > 0,
            ) for i, ip in enumerate(ips)
        ]
        return futures[0].result()


def _babysit(tpu_id, tpu_type, zone, project_id, script=None, stream_log=True):
    """(Re)creates TPU and runs `script`."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    ran_script = False # have we already ran the script on this TPU?

    # create logger for this TPU
    logger = logging.getLogger(tpu_id)
    logger.setLevel(logging.DEBUG)

    # remove previous log handlers
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    
    # add file handler
    log_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(f'logs/{zone}-{tpu_id}.txt', 'w')
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
        
        # check if TPU is healthy
        logger.info('checking TPU status...')
        try:
            # get TPU status
            tpu_info = client.get_queued_resource(name=qr_name)
            tpu_state = tpu_info.state.state.name
            logger.info(f'TPU found, state={tpu_state}')
            
            # if TPU is unhealthy, delete it
            if tpu_state in ('FAILED', 'SUSPENDED'):
                logger.info('deleting TPU...')
                _delete(tpu_id, zone, project_id)
                time.sleep(10)
                raise NotFound('TPU deleted')

        except NotFound as e:
            # if TPU doesn't exist, create it
            logger.info('TPU not found')
            logger.info('creating TPU...')
            _create(tpu_type, tpu_id, zone, project_id)
            ran_script = False


        # if script was provided, wait until TPU is ready, then run the script
        if not ran_script and script is not None:
        
            # wait until TPU is ready
            while True:
                tpu_info = client.get_queued_resource(name=qr_name)
                tpu_state = tpu_info.state.state.name
                logger.info(f'TPU state={tpu_state}')
                if tpu_state == 'ACTIVE': break
                time.sleep(10)
            
            # run script
            logger.info('running script...')
            stdout_writer = LoggerWriter(logger, logging.INFO)
            stderr_writer = LoggerWriter(logger, logging.ERROR)
            result = _run(tpu_id, zone, project_id, script, stdout_writer, stderr_writer)
            logger.info(f'script finished with exit code {result.exited}.')
            ran_script = True

        # wait before checking on the TPU again
        logger.info('sleeping...')
        time.sleep(60)


def babysit(idxs, tpu_type, zone, project_id, script=None):
    """Keeps multiple TPUs alive and running `script`."""

    # crete output directory for logs
    if not os.path.exists('logs'): os.mkdir('logs')

    # create and start a thread for each TPU
    threads = []
    for idx in idxs:
        tpu_id = f'tn-{tpu_type}-{idx}'
        thread = threading.Thread(target=_babysit, args=(tpu_id, tpu_type, zone, project_id, script, False), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(1) # stagger creation
    
    # keep main thread alive while threads are running
    while any(t.is_alive() for t in threads):
        time.sleep(1)
