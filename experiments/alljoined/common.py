"""Constants for the approved THINGS-matched, buffered Alljoined experiment."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from pathlib import Path
import os,sys
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('ALLJOINED_RUN_ROOT',str(REPO/'results/alljoined'))).resolve()
CODE_ROOT=REPO/'experiments/alljoined'
AUDIT=ROOT.parent
PYTHON=os.environ.get('PYTHON', sys.executable)
sys.path.insert(0,str(REPO));sys.path.insert(0,str(AUDIT));sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/numba_cache')
os.environ.setdefault('MPLCONFIGDIR','/tmp/eeg_cross_subject_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
NAME='Alljoined_things_matched_buffered'
def tag(subject):return f'sub-{int(subject):02d}'
def _shared_cuda_owner(pid):
    """Only the specifically authorized Major10 queues may share this slot."""
    try:
        args=[s.decode(errors='replace') for s in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if s]
    except OSError:
        return False
    return any(s in ('experiments.leakfree_things.major10_multiseed','experiments.leakfree_things.major10_train_entry')
               or '/revision_experiments/major10_multiseed/' in s for s in args)

def _gpu_admission(gpu,policy,snapshot):
    """Pure policy function, independent of timing or training outcomes."""
    owners=[int(pid) for pid in snapshot['compute_pids'] if int(pid)!=os.getpid()]
    if not policy.get('enabled',False):
        return (not owners),'exclusive_default'
    if snapshot['free_MiB']<int(policy['minimum_free_MiB']):
        return False,'insufficient_free_memory'
    unknown=[pid for pid in owners if not _shared_cuda_owner(pid)]
    if unknown:
        return False,'another_non_Major10_CUDA_owner:'+','.join(map(str,unknown))
    return True,'authorized_Major10_sharing' if owners else 'available'

def wait_for_gpu_idle(gpu):
    """Historical API: admit authorized sharing with a minimum free-memory reserve."""
    import subprocess,time,json
    while True:
        path=ROOT/'config/gpu_sharing_policy.json'
        policy=load_json(path.read_text()) if path.exists() else {'enabled':False}
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
        values={int(v.split(',')[0]):v.split(',') for v in raw.strip().splitlines()}
        row=values[int(gpu)];uuid=row[1].strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        owners=[int(v.split(',')[1]) for v in apps.strip().splitlines() if v.split(',')[0].strip()==uuid]
        snapshot={'gpu':int(gpu),'uuid':uuid,'free_MiB':int(row[2]),'used_MiB':int(row[3]),'compute_pids':owners}
        admitted,reason=_gpu_admission(gpu,policy,snapshot)
        if admitted:
            event={'time':time.time(),'controller_pid':os.getpid(),'decision':'admitted','reason':reason,**snapshot}
            (ROOT/'status').mkdir(exist_ok=True)
            with (ROOT/'status/gpu_admissions.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print('Physical GPU',gpu,'admitted:',reason,'free MiB',snapshot['free_MiB'],'CUDA owners',owners,flush=True)
            return
        print('Physical GPU',gpu,'waiting:',reason,'free MiB',snapshot['free_MiB'],'CUDA owners',owners,flush=True)
        time.sleep(int(policy.get('poll_seconds',30)))

def cache(subject):
    if int(subject)==1:return AUDIT/'recovery_sub01'
    own=ROOT/'recovery'/tag(subject)
    prior=AUDIT/'matched_image_buffered_v1/recovery'/tag(subject)
    # Raw event recovery is independent of model/data-row ordering. Reuse only
    # a completed verified source cache, never prior fitted models or means.
    if not own.exists() and (prior/'epoch_summary.json').exists():
        import json
        s=load_json((prior/'epoch_summary.json').read_text())
        if s['blocks']==76 and s['official_dropped_disagreements']==0:return prior
    return own

