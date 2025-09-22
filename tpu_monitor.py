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
    for header in ['ID', 'Zone', 'Type', 'Schedule', 'State', 'Created', 'Uptime']:
        table.add_column(header)
    table.caption = f'Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
    
    # add nodes to table
    nodes = client.list_nodes(parent=f'projects/{project_id}/locations/-')
    for node in sorted(nodes, key=lambda x: x.create_time):
        time_created = node.create_time
        time_now = datetime.now(time_created.tzinfo)
        time_last = time_now if not 'maintenance' in node.health_description else parser.parse(node.health_description, fuzzy=True)
        table.add_row(
            node.name.split('/')[-1], # ID
            node.name.split('/')[3], # zone
            node.accelerator_type, # accelerator type
            'spot' if node.scheduling_config.spot else 'on-demand', # spot
            node.state.name, # state
            str(timedelta(seconds=round((time_now - time_created).total_seconds()))).split(',')[0], # created
            str(timedelta(seconds=round((time_last - time_created).total_seconds()))).split(',')[0], # uptime
            style={'READY': 'dark_green', 'CREATING': 'dark_orange', 'PREEMPTED': 'dark_red'}[node.state.name] # color
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
