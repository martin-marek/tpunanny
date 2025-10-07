import sys
import time
import fire
from google.cloud import tpu_v2
from datetime import datetime, timedelta
from dateutil import parser
from rich.live import Live
from rich.table import Table
client = tpu_v2.TpuClient()


def generate_tpu_table(project_id):

    # create empty table
    table = Table()
    for header in ['Request ID', 'Zone', 'Type', 'State', 'Created']:
        table.add_column(header)
    table.caption = f'Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
    
    # add nodes to table
    queued_resources = client.list_queued_resources(parent=f'projects/{project_id}/locations/-')
    for qr in sorted(queued_resources, key=lambda x: x.create_time):
        time_created = qr.create_time
        time_now = datetime.now(time_created.tzinfo)
        node = qr.tpu.node_spec[0].node
        table.add_row(
            qr.name.split('/')[-1], # ID
            qr.name.split('/')[3], # zone
            node.accelerator_type, # accelerator type
            qr.state.state.name, # state
            str(timedelta(seconds=round((time_now - time_created).total_seconds()))).split(',')[0], # created
            style={'ACTIVE': 'dark_green', 'SUSPENDED': 'dark_red'}.get(qr.state.state.name, 'orange1') # color
        )

    return table


def run_monitor(project_id, interval=1):
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
