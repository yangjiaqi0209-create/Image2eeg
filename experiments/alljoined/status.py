"""Read-only queue status; no test inference or checkpoint modification."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import json,re

def main():
    supervisor=ROOT/'status/supervisor.json'
    print('Run:',ROOT)
    print('Supervisor:',supervisor.read_text().strip() if supervisor.exists() else 'not initialized')
    print('\nsubject  stage          state                  last logged epoch')
    for subject in range(1,21):
        p=ROOT/'status'/f'{tag(subject)}.json'
        s=load_json(p.read_text()) if p.exists() else {'stage':'pending','status':'waiting for pilot / worker'}
        stage=s.get('stage','?');last=''
        log=ROOT/'logs'/tag(subject)/f'{stage}.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
            found=re.findall(r'epoch\s+(\d+)\s*(?:\(|:)',tail,re.I)
            if found:last=found[-1]
        if (ROOT/'evaluation'/tag(subject)/'workflow_passed.json').exists():stage='final test';s['status']='completed (no score cutoff)'
        print(f'{tag(subject):9s}{stage:15s}{s.get("status","?"):25s}{last}')

if __name__=='__main__':main()
