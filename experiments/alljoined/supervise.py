"""nohup supervisor: pilot gate, then one serial worker per physical GPU."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import concurrent.futures,datetime,fcntl,json,subprocess,time,traceback
from experiments.alljoined.train_subject import write,event,sha
import argparse

COMPLETED=set()

def verify_completed(subject):
    """Reuse a frozen final evaluation only when both selected model hashes match."""
    dest=ROOT/'evaluation'/tag(subject)
    if not (dest/'metrics.json').exists():return False
    metrics=load_json((dest/'metrics.json').read_text())
    passed=load_json((dest/'workflow_passed.json').read_text())
    claim=load_json((dest/'test_claim.json').read_text())
    assert passed['passed'] and passed['selection_frozen_before_test']
    for stage,key in [('encoder','encoder_sha256'),('stage2','generator_sha256')]:
        selection=load_json((ROOT/'selection'/f'{tag(subject)}_{stage}.json').read_text())
        actual=sha(selection['checkpoint'])
        assert actual==selection['sha256']==metrics[key]==passed[key]==claim[key],f'{tag(subject)} {stage} hash mismatch'
    return True

def wait_gpu(gpu):
    # Use the same explicit sharing policy at supervisor and subject-stage gates.
    return wait_for_gpu_idle(gpu)
def execute(subject,gpu):
    if subject in COMPLETED:
        print('REUSE completed frozen models and final test',tag(subject),flush=True);return True
    wait_gpu(gpu)
    log=ROOT/'logs'/tag(subject)/'pipeline.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        command=[PYTHON,str(CODE_ROOT/'train_subject.py'),'--subject',str(subject),'--gpu',str(gpu)]
        print('START',tag(subject),'physical GPU',gpu,flush=True)
        proc=subprocess.Popen(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
        write(ROOT/'status'/f'{tag(subject)}_pipeline_pid.json',{'pid':proc.pid,'physical_gpu':gpu,'command':command})
        code=proc.wait()
    if code:
        event(subject,'pipeline','failed',exit_code=code,log=str(log));print('FAILED',tag(subject),'exit',code,flush=True);return False
    return True
def final_test(subject,gpu,pilot=False):
    if subject in COMPLETED:return True
    wait_gpu(gpu)
    log=ROOT/'logs'/tag(subject)/'final_test.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=[PYTHON,str(CODE_ROOT/'evaluate.py'),'--subject',str(subject),'--gpu','0']+(['--pilot'] if pilot else [])
    with log.open('a') as f:
        p=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1'})
    if p.returncode:
        event(subject,'final_test','failed',exit_code=p.returncode,log=str(log));return False
    result=ROOT/'evaluation'/tag(subject)/'workflow_passed.json'
    return result.exists() and load_json(result.read_text())['passed']
def worker(subjects,gpu):
    outcomes={}
    for subject in subjects:
        try:outcomes[str(subject)]=execute(subject,gpu)
        except Exception as exc:
            event(subject,'pipeline','failed',error=repr(exc));outcomes[str(subject)]=False;traceback.print_exc()
    return outcomes
def main(resume=False):
    (ROOT/'status').mkdir(exist_ok=True)
    lock=(ROOT/'status/supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if resume:
        COMPLETED.update(s for s in range(1,21) if verify_completed(s))
        record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),
                'reused_completed_subjects':sorted(COMPLETED),'reason':'Resume data recovery after unavailable local download proxy; protocol unchanged'}
        with (ROOT/'status/resume_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print('RESUME; verified completed subjects',sorted(COMPLETED),flush=True)
        for subject in range(1,21):
            if subject not in COMPLETED:event(subject,'queue','pending',reason='Resumed queue; previous failure remains in event history and logs')
    write(ROOT/'status/supervisor.json',{'pid':os.getpid(),'phase':'pilot_training','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol':'THINGS matched + explicitly approved train/val buffer; all 20 subjects, no test-performance screening'})
    for filename in ['train_subject.py','recover.py','prepare.py','evaluate.py','summarize.py']:
        assert (ROOT/filename).is_file()
    if not execute(1,0):
        write(ROOT/'status/pilot_gate.json',{'passed':False,'reason':'training/data failure; logs preserved; no other subjects launched'});return
    if not final_test(1,0,True):
        write(ROOT/'status/pilot_gate.json',{'passed':False,'reason':'structural/numerical evaluation failure; no accuracy threshold applied; other subjects not launched'});return
    write(ROOT/'status/pilot_gate.json',{'passed':True,'performance_threshold_applied':False,'subject':'sub-01','auto_expand_to_others':True})
    write(ROOT/'status/supervisor.json',{'pid':os.getpid(),'phase':'other_subjects_training','pilot_passed':True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        a=executor.submit(worker,list(range(2,21,2)),0)
        b=executor.submit(worker,list(range(3,21,2)),1)
        outcomes={**a.result(),**b.result()}
    write(ROOT/'status/all_training_outcomes.json',outcomes)
    # Remaining official tests only after all queued model-selection attempts.
    write(ROOT/'status/supervisor.json',{'pid':os.getpid(),'phase':'final_evaluation','pilot_passed':True,'training_outcomes':outcomes})
    def evaluate_worker(subjects,gpu):
        for subject in subjects:
            if outcomes.get(str(subject)):
                try:final_test(subject,gpu)
                except Exception as exc:event(subject,'final_test','failed',error=repr(exc));traceback.print_exc()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f=[executor.submit(evaluate_worker,list(range(2,21,2)),0),executor.submit(evaluate_worker,list(range(3,21,2)),1)]
        for task in f:task.result()
    result=subprocess.run([PYTHON,str(CODE_ROOT/'summarize.py')],cwd=REPO,env=os.environ.copy())
    n=sum((ROOT/'evaluation'/tag(s)/'metrics.json').exists() for s in range(1,21))
    write(ROOT/'status/supervisor.json',{'pid':os.getpid(),'phase':'complete' if n==20 and result.returncode==0 else 'finished_with_recorded_failures',
        'subjects_with_metrics':n,'summary_exit_code':result.returncode,'training_outcomes':outcomes})
    print('QUEUE FINISHED; metrics',n,'/20; no low-performance subjects excluded',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:main(args.resume)
    except Exception:
        traceback.print_exc();write(ROOT/'status/supervisor_error.json',{'error':traceback.format_exc(),'time':time.time()});raise

