import re
import sys
import time
import fire
from google.cloud import tpu_v2
from google.api_core import exceptions
from datetime import datetime, timedelta
from rich.live import Live
from rich.table import Table

client = tpu_v2.TpuClient()

TEXT_COLOR = {
    'ACTIVE': 'dark_green',
    'PROVISIONING': 'dodger_blue2',
    'WAITING_FOR_RESOURCES': 'dodger_blue2',
    'SUSPENDING': 'dark_red',
    'SUSPENDED': 'dark_red',
    'FAILED': 'dark_red',
    'OTHER': 'grey50'
}

def natsort_key(s):
    parts = re.split(r'(\d+)', s)
    parts[1::2] = map(int, parts[1::2])
    return parts

def generate_tpu_table(project_id, timeout=10):
    table = Table()
    for header in ['Request ID', 'Zone', 'Type', 'IP', 'State', 'Created']:
        table.add_column(header)
    table.caption = f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    try:
        # Fetch all resources once to avoid N+1 query problem
        queued_resources = client.list_queued_resources(parent=f'projects/{project_id}/locations/-', timeout=timeout)
        all_nodes = list(client.list_nodes(parent=f'projects/{project_id}/locations/-', timeout=timeout))
        
        # Create lookup map for faster access
        node_map = {node.name: node for node in all_nodes}

        for qr in sorted(queued_resources, key=lambda qr: natsort_key(qr.name.split('/')[-1])):
            zone = qr.name.split('/')[3]
            qr_id = qr.name.split('/')[-1]
            
            # Calculate elapsed time
            time_created = qr.create_time
            time_now = datetime.now(time_created.tzinfo)
            time_elapsed_str = str(timedelta(seconds=round((time_now - time_created).total_seconds()))).split(',')[0]

            # Resolve IP from pre-fetched node map
            ip = ''
            try:
                tpu_id = qr.tpu.node_spec[0].node_id
                tpu_name = f'projects/{project_id}/locations/{zone}/nodes/{tpu_id}'
                
                if tpu_name in node_map:
                    endpoints = node_map[tpu_name].network_endpoints
                    if endpoints:
                        ip = sorted([ep.access_config.external_ip for ep in endpoints])[0]
            except Exception:
                pass

            table.add_row(
                qr_id,
                zone,
                qr.tpu.node_spec[0].node.accelerator_type,
                ip,
                qr.state.state.name,
                time_elapsed_str,
                style=TEXT_COLOR.get(qr.state.state.name, TEXT_COLOR['OTHER'])
            )

    except Exception as e:
        error_type = type(e).__name__
        table.caption = f'⚠️ Connection ({error_type}). Retrying... Last attempt: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

    return table

def run_monitor(project_id, interval=3):
    """Monitors TPUs in a GCP project with a live-updating table."""
    try:
        with Live(generate_tpu_table(project_id), screen=True, auto_refresh=False) as live:
            while True:
                live.update(generate_tpu_table(project_id), refresh=True)
                time.sleep(interval)
    except KeyboardInterrupt:
        print('\n👋 Monitoring stopped.')
        sys.exit(0)

if __name__ == '__main__':
    fire.Fire(run_monitor)
