"""Scheduling-only handoff for the authorized Major10/Alljoined shared GPUs.

Preserves active raw recovery, reuses selected models, and delegates all actual
training/evaluation to the original frozen workflow. No data check is relaxed.
"""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,datetime,json,signal,time,traceback
from experiments.alljoined import supervise as original
from experiments.alljoined.train_subject import sha,write,event

BASE_EXECUTE=original.execute
HANDOFF=ROOT/'status/gpu_sharing_patch_20261113/handoff.json'

def alive(pid):
    p=Path('/proc')/str(pid)
    try:
        state=next(s.split()[1] for s in (p/'status').read_text().splitlines() if s.startswith('State:'))
        return state!='Z'
    except (OSError,StopIteration):return False

def direct_children(pid):
    try:return [int(v) for v in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
    except OSError:return []

def args(pid):
    try:return [s.decode(errors='replace') for s in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if s]
    except OSError:return []

def trained(subject):
    markers=[ROOT/'status'/f'{tag(subject)}_{stage}_success.json' for stage in ['encoder','stage1','stage2']]
    if not all(p.exists() for p in markers):return False
    for stage in ['encoder','stage1','stage2']:
        p=ROOT/'selection'/f'{tag(subject)}_{stage}.json'
        if not p.exists():raise RuntimeError('Completed training is missing a selection record: '+str(p))
        s=load_json(p.read_text());assert sha(s['checkpoint'])==s['sha256']
        if stage!='encoder':assert not s.get('selection_used_test',False) and s['args']['skip_test']
    return True

def adopt_until_safe(subject,pid):
    """Let existing CPU data work finish; replace only the old idle controller."""
    original_args=args(pid)
    if not alive(pid):return
    assert str(CODE_ROOT/'train_subject.py') in original_args
    assert original_args[original_args.index('--subject')+1]==str(subject)
    while alive(pid):
        if trained(subject):
            print('ADOPT existing training completed',tag(subject),pid,flush=True)
            # Selection records are committed before the controller exits.
            while alive(pid):time.sleep(2)
            return
        checks=ROOT/'status'/f'{tag(subject)}_data_checks_success.json'
        if checks.exists():
            os.kill(pid,signal.SIGSTOP)
            try:
                children=[p for p in direct_children(pid) if alive(p)]
                if children:
                    print('ADOPT preserves active subprocesses',tag(subject),children,flush=True)
                else:
                    # No active training or recovery is stopped. Only the obsolete idle controller exits.
                    os.kill(pid,signal.SIGTERM)
                    os.kill(pid,signal.SIGCONT)
                    for _ in range(100):
                        if not alive(pid):break
                        time.sleep(.1)
                    if alive(pid):raise RuntimeError('Old idle controller did not exit; no replacement launched')
                    record={'time':time.time(),'subject':tag(subject),'old_controller_pid':pid,
                            'reason':'Data checks complete; no active child; replace obsolete exclusive wait with shared-memory scheduler',
                            'active_training_interrupted':False,'active_recovery_interrupted':False}
                    write(ROOT/'status'/f'{tag(subject)}_scheduling_handoff.json',record)
                    print('REPLACED idle controller',tag(subject),pid,flush=True)
                    return
            finally:
                if alive(pid):os.kill(pid,signal.SIGCONT)
        print('ADOPT waiting for existing data recovery/preparation',tag(subject),'controller',pid,'children',direct_children(pid),flush=True)
        time.sleep(10)
    # If the old controller failed in data validation, preserve that failure rather than bypass it.
    status=ROOT/'status'/f'{tag(subject)}.json'
    if status.exists():
        s=load_json(status.read_text())
        if s.get('status')=='failed':raise RuntimeError('Adopted pipeline failed; unchanged data validation remains blocking: '+str(s))

def execute(subject,gpu):
    if subject in original.COMPLETED:
        print('REUSE existing verified final test',tag(subject),flush=True);return True
    if trained(subject):
        print('REUSE existing selected src/encoder/stage1/stage2, no retraining',tag(subject),flush=True);return True
    handoff=load_json(HANDOFF.read_text())
    adopted=handoff.get('adopt_controllers',{}).get(str(subject))
    if adopted:
        adopt_until_safe(subject,int(adopted))
        if trained(subject):return True
    # The known subject-08 event disagreement is a data issue, not a scheduling issue.
    if subject==8:
        p=ROOT/'logs/sub-08/recover.log'
        resolved=False
        proof=ROOT/'recovery/sub-08/boundary_resolution.json'
        independent=ROOT/'preprocessing/sub-08/independent_data_checks.json'
        if proof.exists() and independent.exists():
            from experiments.alljoined.sub08_boundary_repair import validate_recovery_summary
            source=ROOT/'recovery/sub-08'
            resolved=bool(validate_recovery_summary(source,load_json((source/'epoch_summary.json').read_text()))) and load_json(independent.read_text())['passed']
        if p.exists() and 'official dropped disagreements' in p.read_text(errors='replace') and not resolved:
            event(subject,'data_validation','blocked',reason='Existing official dropped-flag mismatch is unresolved; scheduling patch does not bypass or rerun it')
            print('DATA BLOCK remains',tag(subject),'official dropped disagreements; no threshold relaxed',flush=True)
            return False
    return BASE_EXECUTE(subject,gpu)

original.execute=execute

def main():
    manifest=load_json(HANDOFF.read_text())
    print('SHARED GPU scheduling handoff',json.dumps(manifest),flush=True)
    original.main(resume=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'status/shared_supervisor_error.json',{'time':time.time(),'error':traceback.format_exc()})
        raise
