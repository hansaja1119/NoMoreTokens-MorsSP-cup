"""Compact read-only status of local experiments."""
import json
from common import SCRIPTS

def main():
    queue=SCRIPTS/'runs'/'experiment_queue'/'status.json'
    if queue.exists():
        data=json.loads(queue.read_text());print('Queue:',data['state'],data.get('stage',data.get('selected_run','')))
        if data.get('error'):print('Error:',data['error'])
    for path in sorted((SCRIPTS/'runs').glob('*/status.json')):
        if path.parent.name=='experiment_queue':continue
        state=json.loads(path.read_text())
        print(path.parent.name,'|',state['state'],'| step',state.get('step','-'),'| best validation',state.get('best_score','-'))

if __name__=='__main__':main()
