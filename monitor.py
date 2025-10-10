import re
import sys
import time
import fire
from google.cloud import tpu_v2
from google.api_core import exceptions
from datetime import datetime, timedelta
from dateutil import parser
from rich.live import Live
from rich.table import Table
client = tpu_v2.TpuClient()


def natsort_key(s):
    # re.split('(\d+)', s) splits the string by digits, keeping the digits.
    # e.g., 'v6e-16' becomes ['v', '6', 'e-', '16', '']
    parts = re.split(r'(\d+)', s)
    # Convert numeric parts to integers for correct numerical sorting
    parts[1::2] = map(int, parts[1::2])
    return parts


def generate_tpu_table(project_id):

    # create empty table
    table = Table()
    for header in ['Request ID', 'Zone', 'Type', 'State', 'Created']:
        table.add_column(header)
    table.caption = f'Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
    
    # add nodes to table
    try:
        text_color = {
            'ACTIVE': 'dark_green',
            'PROVISIONING': 'dodger_blue2',
            'WAITING_FOR_RESOURCES': 'dodger_blue2',
            'SUSPENDING': 'dark_red',
            'SUSPENDED': 'dark_red',
            'FAILED': 'dark_red',
            'OTHER': 'grey50'
        }
        queued_resources = client.list_queued_resources(parent=f'projects/{project_id}/locations/-')
        for qr in sorted(queued_resources, key=lambda qr: natsort_key(qr.name.split('/')[-1])): # sort by name
            time_created = qr.create_time
            time_now = datetime.now(time_created.tzinfo)
            node = qr.tpu.node_spec[0].node
            table.add_row(
                qr.name.split('/')[-1], # request ID
                qr.name.split('/')[3], # zone
                node.accelerator_type, # accelerator type
                qr.state.state.name, # state
                str(timedelta(seconds=round((time_now - time_created).total_seconds()))).split(',')[0], # time created
                style=text_color.get(qr.state.state.name, text_color['OTHER']) # text color
            )
    except exceptions.ServiceUnavailable:
        table.caption = f'⚠️ Connection error. Retrying... Last attempt: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


    return table


def run_monitor(project_id, interval=10):
    """Monitors TPUs in a GCP project with a live-updating table."""
    try:
        with Live(generate_tpu_table(project_id), screen=True, auto_refresh=False) as live:
            while True:
                time.sleep(interval)
                live.update(generate_tpu_table(project_id), refresh=True)
    except KeyboardInterrupt:
        print('\n👋 Monitoring stopped.')
        sys.exit(0)


if __name__ == '__main__':
    fire.Fire(run_monitor)
