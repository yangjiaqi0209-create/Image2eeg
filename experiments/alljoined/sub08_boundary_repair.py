"""Resolve a pinned, physically truncated EDF boundary without accepting bad data.

Only exact rows frozen by a complete, hash-checked sub-08 acquisition audit may
disagree with official labels. Their complete epochs cross verified EDF edges;
official-style concatenation independently rejects the same events. Original
official metadata and the actual nonzero disagreement count are preserved.
"""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,datetime,fcntl,gc,hashlib,importlib.util,json,subprocess,time,warnings
import mne,numpy as np,pandas as pd
from validate_sub01 import align

EVIDENCE=ROOT/'status/sub08_validation_repair'
EXPECTED_ROWS=set(range(6311,6316))
EXPECTED_IMAGES=[9875,1997,8833,5072,3843]
EXPECTED_SAMPLES=[78855,78907,78959,79007,79055]
EXPECTED_EDF_SHA='4ff997729bbed296b9ebf1f213216016a381618dca512972a9070a848e9729cd'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def audit_all_boundaries():
    """Freeze a complete source-only audit before admitting any exception."""
    source=ROOT/'recovery/sub-08'
    manifest=load_json((source/'raw_download_manifest.json').read_text())
    assert len(manifest)==76
    for f in manifest:
        assert digest(source/f['path'])==f['sha256']
    metadata=source/'official/preprocessed_eeg/sub-08/experiment_metadata_categories.parquet'
    d=pd.read_parquet(metadata).reset_index(names='official_index_label');d['metadata_row']=np.arange(len(d))
    d['image_id']=d.image_path.map(lambda p:int(Path(p).stem))
    spec=importlib.util.spec_from_file_location('sub08_full_boundary_audit',source/'upstream/preprocessing_utils.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    conflicts=[];blocks=[]
    for session in range(1,5):
        for block in range(1,20):
            path=next((source/f'official/raw_eeg/sub-08/session_{session:02d}/block_{block:02d}').glob('*.edf'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');raw=module.read_eeg_data(str(path.parent))
            meta=d[(d.session==session)&(d.block_id==block)&(d.partition!='oddball')].copy().reset_index(drop=True)
            ev,ids=mne.events_from_annotations(raw,regexp='.*stim',verbose=False)
            reverse={v:k for k,v in ids.items()};tokens=[(reverse[int(v)].split(',')[0],int(reverse[int(v)].split(',')[1])) for v in ev[:,2]]
            positions,method=align(tokens,meta)
            ep=mne.Epochs(raw,ev,event_id=ids,tmin=-.2,tmax=1,baseline=(None,0),preload=True,reject=None,event_repeated='drop',verbose=False)
            valid=np.zeros(len(meta),bool);valid[positions[ep.selection]]=True
            mismatch=np.flatnonzero(valid!=~meta.dropped.to_numpy())
            for row in mismatch:
                event=np.flatnonzero(positions==row)
                assert len(event)==1
                event=int(event[0]);sample=int(ev[event,0]);sf=raw.info['sfreq']
                lo=sample+round(-.2*sf);hi=sample+round(sf)
                assert not valid[row] and not meta.iloc[row].dropped,'Non-boundary conflict requires separate investigation'
                assert ep.drop_log[event] in [('TOO_SHORT',),('NO_DATA',)]
                assert lo<0 or hi>=raw.n_times,'Complete epoch unexpectedly rejected'
                record=dict(metadata_row=int(meta.iloc[row].metadata_row),official_index_label=int(meta.iloc[row].official_index_label),
                    session=session,block=block,image_id=int(meta.iloc[row].image_id),sample=sample,
                    n_samples=int(raw.n_times),sampling_hz=float(sf),requested_first_sample=lo,requested_last_sample=hi,
                    missing_samples=max(0,-lo)+max(0,hi-(raw.n_times-1)),drop_reason=ep.drop_log[event][0],
                    official_dropped=False,computed_valid=False,edf_sha256=digest(path),edf_file=str(path.relative_to(source)))
                conflicts.append(record)
            if len(mismatch):
                rejected=[int(ev[i,0]) for i,reason in enumerate(ep.drop_log) if reason]
                verify_concatenation(source,module,session,block,rejected,len(ep))
            blocks.append(dict(session=session,block=block,valid_epochs=len(ep),official_disagreements=len(mismatch),
                              edf_sha256=digest(path),raw_samples=raw.n_times))
            save(EVIDENCE/'boundary_audit_progress.json',dict(blocks_checked=len(blocks),conflicts=len(conflicts),latest=blocks[-1]))
            print('AUDIT sub08',session,block,'conflicts',len(mismatch),'total',len(conflicts),flush=True)
            del raw,ep;gc.collect()
    assert len(blocks)==76 and EXPECTED_ROWS<=set(r['metadata_row'] for r in conflicts)
    pd.DataFrame(conflicts).to_csv(EVIDENCE/'boundary_audit_trials.csv',index=False)
    save(EVIDENCE/'boundary_audit.json',dict(passed=True,subject=8,blocks_checked=76,
         official_metadata_sha256=digest(metadata),resolved_disagreements=len(conflicts),unresolved_disagreements=0,
         metadata_rows=sorted(r['metadata_row'] for r in conflicts),records=conflicts,blocks=blocks,
         rule='Only source-verified incomplete fixed-window epochs; no test predictions, no threshold change, no row relabelled valid'))

def validate_recovery_summary(source,summary):
    """Strict default; exact, hash-bound erratum is the sole allowed exception."""
    if summary['blocks']!=76:raise RuntimeError('Expected all76 acquisition blocks')
    if int(summary.get('unresolved_official_dropped_disagreements',0)):
        raise RuntimeError('Unresolved discrepancy cannot be admitted')
    count=int(summary['official_dropped_disagreements'])
    if not count:return None
    if Path(source).name!='sub-08':raise RuntimeError('Unresolved official dropped disagreement')
    audit=load_json((EVIDENCE/'boundary_audit.json').read_text())
    assert audit['passed'] and audit['subject']==8 and audit['blocks_checked']==76
    expected=set(audit['metadata_rows']);assert EXPECTED_ROWS<=expected
    assert count==len(expected)==audit['resolved_disagreements']
    record=Path(source)/'boundary_resolution.json'
    proof=load_json(record.read_text())
    assert proof['passed'] and proof['resolved_disagreements']==count and proof['unresolved_disagreements']==0
    assert set(proof['metadata_rows'])==expected
    assert proof['boundary_audit_sha256']==digest(EVIDENCE/'boundary_audit.json')
    assert proof['official_metadata_sha256']==digest(Path(source)/'official/preprocessed_eeg/sub-08/experiment_metadata_categories.parquet')
    assert audit['official_metadata_sha256']==proof['official_metadata_sha256']
    ledger=pd.read_parquet(Path(source)/'trial_ledger.parquet')
    assert digest(Path(source)/'trial_ledger.parquet')==proof['trial_ledger_sha256']
    mismatch=ledger[ledger.valid.to_numpy()!=~ledger.dropped.to_numpy()]
    assert set(mismatch.metadata_row)==expected and len(mismatch)==count
    assert not mismatch.valid.any() and not mismatch.dropped.any()
    for r in audit['records']:
        row=mismatch[mismatch.metadata_row==r['metadata_row']].iloc[0]
        assert (row.session,row.block_id,row.image_id,row.edf_sample_256hz,row.exclusion_reason)==(r['session'],r['block'],r['image_id'],r['sample'],r['drop_reason'])
    assert summary.get('resolved_official_dropped_disagreements')==count
    assert summary.get('unresolved_official_dropped_disagreements')==0
    return proof

def recover():
    source=ROOT/'recovery/sub-08'
    if (source/'epoch_summary.json').exists():
        validate_recovery_summary(source,load_json((source/'epoch_summary.json').read_text()));return
    manifest=load_json((source/'raw_download_manifest.json').read_text())
    assert len(manifest)==76
    for f in manifest:
        path=source/f['path']
        assert path.stat().st_size==f['bytes'] and digest(path)==f['sha256']
    metadata=source/'official/preprocessed_eeg/sub-08/experiment_metadata_categories.parquet'
    metadata_sha=digest(metadata)
    audit=load_json((EVIDENCE/'boundary_audit.json').read_text())
    assert audit['passed'] and audit['blocks_checked']==76 and audit['official_metadata_sha256']==metadata_sha
    expected_rows=set(audit['metadata_rows']);count=len(expected_rows)
    assert EXPECTED_ROWS<=expected_rows and count==audit['resolved_disagreements']
    d=pd.read_parquet(metadata)
    stim=pd.read_parquet(source/'official/preprocessed_eeg/sub-08/stim_order.parquet')
    assert stim.equals(d[stim.columns]) and d.index.is_unique
    d=d.reset_index(names='official_index_label');d['metadata_row']=np.arange(len(d))
    d['image_id']=d.image_path.map(lambda p:int(Path(p).stem))
    spec=importlib.util.spec_from_file_location('sub08_pinned_preprocessor',source/'upstream/preprocessing_utils.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    records=[];ledgers=[];exceptions=[]
    for session in range(1,5):
        for block in range(1,20):
            stem=f'ses-{session:02d}_block-{block:02d}'
            info_path=source/'epochs'/f'{stem}.json';ledger_path=source/'epochs'/f'{stem}.parquet'
            if info_path.exists() and ledger_path.exists():
                info=load_json(info_path.read_text());ledger=pd.read_parquet(ledger_path)
                if info['official_dropped_disagreements']:
                    mism=ledger[ledger.valid.to_numpy()!=~ledger.dropped.to_numpy()]
                    expected_block={r['metadata_row'] for r in audit['records'] if (r['session'],r['block'])==(session,block)}
                    assert set(mism.metadata_row)==expected_block
                    assert not mism.valid.any() and not mism.dropped.any()
                    assert load_json((EVIDENCE/f'concatenation_ses{session:02d}_block{block:02d}.json').read_text())['passed']
                else:
                    assert np.array_equal(ledger.valid.to_numpy(),~ledger.dropped.to_numpy())
                records.append(info);ledgers.append(ledger);continue
            path=next((source/f'official/raw_eeg/sub-08/session_{session:02d}/block_{block:02d}').glob('*.edf'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');raw=module.read_eeg_data(str(path.parent))
            meta=d[(d.session==session)&(d.block_id==block)&(d.partition!='oddball')].copy().reset_index(drop=True)
            events,ids=mne.events_from_annotations(raw,regexp='.*stim',verbose=False)
            reverse={v:k for k,v in ids.items()};descriptions=[reverse[int(v)] for v in events[:,2]]
            tokens=[(v.split(',')[0],int(v.split(',')[1])) for v in descriptions]
            positions,method=align(tokens,meta)
            ep=mne.Epochs(raw,events,event_id=ids,tmin=-.2,tmax=1.,baseline=(None,0),preload=True,
                          reject=None,event_repeated='drop',verbose=False)
            meta['edf_annotation_event_index']=-1;meta['edf_sample_256hz']=-1
            meta['annotation_description']='';meta['epoch_index_in_file']=-1
            meta['valid']=False;meta['exclusion_reason']='missing_annotation'
            meta['epoch_file']=f'epochs/{stem}.npy';meta['validity_resolution']=''
            for index,position in enumerate(positions):
                meta.loc[position,['edf_annotation_event_index','edf_sample_256hz','annotation_description']]=[index,int(events[index,0]),descriptions[index]]
                meta.loc[position,'exclusion_reason']='|'.join(ep.drop_log[index]) or 'kept'
            keep=positions[ep.selection];meta.loc[keep,'valid']=True
            meta.loc[keep,'epoch_index_in_file']=np.arange(len(ep))
            mismatch=meta[meta.valid.to_numpy()!=~meta.dropped.to_numpy()].copy()
            expected_block=[r for r in audit['records'] if (r['session'],r['block'])==(session,block)]
            assert set(mismatch.metadata_row)=={r['metadata_row'] for r in expected_block},f'Unfrozen mismatch {stem}'
            if len(mismatch):
                assert not mismatch.valid.any() and not mismatch.dropped.any()
                for expected in expected_block:
                    row=mismatch[mismatch.metadata_row==expected['metadata_row']].iloc[0]
                    assert row.image_id==expected['image_id'] and row.edf_sample_256hz==expected['sample']
                    assert row.exclusion_reason==expected['drop_reason']
                    assert raw.n_times==expected['n_samples'] and digest(path)==expected['edf_sha256']
                    assert expected['requested_first_sample']<0 or expected['requested_last_sample']>=raw.n_times
                meta.loc[mismatch.index,'validity_resolution']='verified_incomplete_EDF_boundary; excluded'
                exceptions.append(meta.loc[mismatch.index].copy())
                assert load_json((EVIDENCE/f'concatenation_ses{session:02d}_block{block:02d}.json').read_text())['passed']
            if ep.info['sfreq']!=250:ep.resample(250,verbose=False)
            data=ep.get_data().astype(np.float32)
            assert np.isfinite(data).all() and data.shape[1:]==(32,301)
            assert np.array_equal(meta.loc[keep].image_id.to_numpy(),[tokens[i][1] for i in ep.selection])
            np.save(source/'epochs'/f'{stem}.npy',data)
            meta.to_parquet(ledger_path,index=False)
            info=dict(session=session,block=block,edf_file=str(path.relative_to(source/'official')),
                annotation_alignment=method,valid_epochs=len(ep),official_dropped_disagreements=len(mismatch),
                resolved_official_dropped_disagreements=len(mismatch),unresolved_official_dropped_disagreements=0,
                channels=ep.ch_names,times=ep.times.tolist(),recording_duration_s=raw.n_times/raw.info['sfreq'],
                meas_date=raw.info['meas_date'].isoformat() if raw.info['meas_date'] else None)
            save(info_path,info);records.append(info);ledgers.append(meta)
            print('sub-08',stem,'verified',len(ep),'documented boundary conflicts',len(mismatch),flush=True)
            del raw,ep,data;gc.collect()
    assert len(records)==76
    ledger=pd.concat(ledgers,ignore_index=True).sort_values('metadata_row')
    assert not ledger.metadata_row.duplicated().any()
    assert set(ledger.metadata_row)==set(d.loc[d.partition!='oddball','metadata_row'])
    assert all(r['channels']==records[0]['channels'] and np.array_equal(r['times'],records[0]['times']) for r in records)
    ledger.to_parquet(source/'trial_ledger.parquet',index=False)
    d[d.partition=='oddball'].to_parquet(source/'excluded_oddball.parquet',index=False)
    mismatch=ledger[ledger.valid.to_numpy()!=~ledger.dropped.to_numpy()]
    assert set(mismatch.metadata_row)==expected_rows
    mismatch.to_csv(EVIDENCE/'resolved_trial_conflicts.csv',index=False)
    proof=dict(passed=True,resolved_disagreements=count,unresolved_disagreements=0,
        metadata_rows=sorted(expected_rows),boundary_audit_sha256=digest(EVIDENCE/'boundary_audit.json'),
        official_metadata_sha256=metadata_sha,trial_ledger_sha256=digest(source/'trial_ledger.parquet'),
        metadata_unchanged=digest(metadata)==metadata_sha,excluded_rows_not_relabelled_as_valid=True,
        complete_blocks=76,scope='sub-08 only; exact hash-bound acquisition boundaries from full76-block audit')
    save(source/'boundary_resolution.json',proof)
    summary=dict(subject='sub-08',blocks=76,valid=int(ledger.valid.sum()),official_dropped_disagreements=count,
        resolved_official_dropped_disagreements=count,unresolved_official_dropped_disagreements=0,
        boundary_resolution=str(source/'boundary_resolution.json'),channels=records[0]['channels'],times=records[0]['times'],blocks_detail=records)
    save(source/'epoch_summary.json',summary);validate_recovery_summary(source,summary)
    save(EVIDENCE/'validation_passed.json',proof)
    print('SUB-08 RAW RECOVERY VALIDATED: all76blocks;',count,'documented incomplete epochs excluded; zero unresolved discrepancies',flush=True)

def verify_concatenation(source,module,session=1,block=6,expected_samples=None,expected_kept=1105):
    raws=[];lengths=[]
    neighbors=[b for b in [block-1,block,block+1] if (1<=b<=4 if block<=4 else 5<=b<=19)]
    for b in neighbors:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');r=module.read_eeg_data(str(source/f'official/raw_eeg/sub-08/session_{session:02d}/block_{b:02d}'))
        ann=r.annotations
        r.set_annotations(mne.Annotations(ann.onset,ann.duration,[f'session_{session},block_{b},{s}' for s in ann.description],orig_time=ann.orig_time))
        lengths.append(r.n_times);raws.append(r)
    r=mne.concatenate_raws(raws,verbose=False)
    ev,ids=mne.events_from_annotations(r,regexp='.*stim',verbose=False)
    keys={k:v for k,v in ids.items() if k.startswith(f'session_{session},block_{block},')}
    ep=mne.Epochs(r,ev,event_id=keys,tmin=-.2,tmax=1,baseline=(None,0),preload=True,reject=None,event_repeated='drop',verbose=False)
    offset=sum(lengths[:neighbors.index(block)])
    rejected=[int(sample-offset) for index,(sample,_,eid) in enumerate(ev) if eid in keys.values() and ep.drop_log[index]]
    reasons=[ep.drop_log[index] for index,(_,_,eid) in enumerate(ev) if eid in keys.values() and ep.drop_log[index]]
    expected_samples=EXPECTED_SAMPLES if expected_samples is None else expected_samples
    assert rejected==expected_samples and all(x in [('BAD boundary',),('TOO_SHORT',),('NO_DATA',)] for x in reasons) and len(ep)==expected_kept
    save(EVIDENCE/f'concatenation_ses{session:02d}_block{block:02d}.json',dict(passed=True,kept=len(ep),rejected_samples=rejected,drop_reasons=reasons,
        method='same pinned upstream concatenate_raws and default MNE Epochs boundary rejection'))

def controller():
    from experiments.alljoined.train_subject import run,setup,train,event
    setup();env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    save(EVIDENCE/'controller_state.json',dict(pid=os.getpid(),state='recovering',time=time.time()))
    run(8,'recover',[PYTHON,str(Path(__file__).resolve()),'--recover'],env)
    run(8,'prepare',[PYTHON,str(CODE_ROOT/'prepare.py'),'--subject','8'],env)
    run(8,'data_checks',[PYTHON,str(CODE_ROOT/'check_data.py'),'--subject','8'],env)
    # Let each existing card finish its assigned training queue. Do not steal
    # a gap between another subject's encoder, Stage1 and Stage2.
    while True:
        ready=[gpu for gpu,tail in [(0,20),(1,19)] if (ROOT/'status'/f'{tag(tail)}_stage2_success.json').exists()]
        if ready:gpu=ready[0];break
        event(8,'queue','pending',reason='Boundary validation repaired; waiting for an existing Alljoined GPU training queue to finish')
        save(EVIDENCE/'controller_state.json',dict(pid=os.getpid(),state='validated_waiting_for_training_slot',time=time.time()))
        time.sleep(30)
    save(EVIDENCE/'controller_state.json',dict(pid=os.getpid(),state='training',gpu=gpu,time=time.time()))
    train(8,gpu)
    from experiments.alljoined import supervise
    assert supervise.final_test(8,gpu)
    # Avoid racing the existing supervisor's final evaluation / status commit.
    with (ROOT/'status/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        outcomes_path=ROOT/'status/all_training_outcomes.json'
        outcomes=load_json(outcomes_path.read_text()) if outcomes_path.exists() else {}
        outcomes['8']=True;save(outcomes_path,outcomes)
        result=subprocess.run([PYTHON,str(CODE_ROOT/'summarize.py')],cwd=REPO,env=env)
        count=sum((ROOT/'evaluation'/tag(s)/'metrics.json').exists() for s in range(1,21))
        save(ROOT/'status/supervisor.json',dict(pid=os.getpid(),phase='complete' if count==20 and result.returncode==0 else 'finished_with_recorded_failures',
            subjects_with_metrics=count,summary_exit_code=result.returncode,training_outcomes=outcomes,sub08_boundary_repair=True))
    save(EVIDENCE/'controller_state.json',dict(pid=os.getpid(),state='complete',subjects_with_metrics=count,time=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recover',action='store_true');parser.add_argument('--audit-all',action='store_true');args=parser.parse_args()
    try:
        if args.audit_all:audit_all_boundaries()
        elif args.recover:recover()
        else:
            EVIDENCE.mkdir(parents=True,exist_ok=True)
            with (EVIDENCE/'controller.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                controller()
    except Exception:
        import traceback
        save(EVIDENCE/'controller_error.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
