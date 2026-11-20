"""Recover one subject using pinned official sources and verified event rules."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,importlib.util,json,re,warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import mne,numpy as np,pandas as pd
import recover_sub01_sources as remote
from validate_sub01 import align
import time,requests

def enable_network_wait():
    """Retry transport/outage errors, never relax checksums or event checks."""
    original=remote.requests.get
    def resilient_get(*args,**kwargs):
        attempt=0
        while True:
            try:
                response=original(*args,**kwargs)
                if response.status_code not in (408,429,500,502,503,504):return response
                code=response.status_code;response.close()
                reason=f'HTTP {code}'
            except (requests.exceptions.ConnectionError,requests.exceptions.Timeout) as exc:
                reason=type(exc).__name__
            attempt+=1;delay=min(300,15*attempt)
            print(f'NETWORK WAIT: {reason}; retry {attempt} in {delay}s; subject remains included',flush=True)
            time.sleep(delay)
    remote.requests.get=resilient_get

def main(subject):
    enable_network_wait()
    if subject==1:
        assert (cache(1)/'trial_ledger.parquet').exists();return
    dest=cache(subject);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'epoch_summary.json').exists():
        summary=load_json((dest/'epoch_summary.json').read_text())
        if summary['official_dropped_disagreements']:
            from experiments.alljoined.sub08_boundary_repair import validate_recovery_summary
            validate_recovery_summary(dest,summary)
        return
    remote.ROOT=dest
    inventory=load_json((AUDIT/'recovery_sub01/resource_inventory.json').read_text())
    revision=inventory['huggingface_revision']
    (dest/'upstream').symlink_to(AUDIT/'recovery_sub01/upstream',target_is_directory=True) if not (dest/'upstream').exists() else None
    globalmeta=dest/'official/preprocessed_eeg/experiment_metadata_categories.parquet';globalmeta.parent.mkdir(parents=True,exist_ok=True)
    if not globalmeta.exists():globalmeta.symlink_to(AUDIT/'recovery_sub01/official/preprocessed_eeg/experiment_metadata_categories.parquet')
    for name in ['stim_order.parquet','experiment_metadata_categories.parquet']:
        rel=f'preprocessed_eeg/{tag(subject)}/{name}'
        remote.fetch(f'{remote.HF}/datasets/{remote.REPO}/resolve/{revision}/{rel}',dest/'official'/rel)
    trees=remote.tree(f'raw_eeg/{tag(subject)}',revision)
    (dest/'resource_inventory.json').write_text(json.dumps({'revision':revision,'files':trees},indent=2)+'\n')
    # The official tree also contains nested backup directories (e.g.
    # sub-07/session_01/session_02). Use the canonical subject/session/block
    # hierarchy audited in source_availability.json, never those duplicates.
    canonical=rf'^raw_eeg/{tag(subject)}/session_0[1-4]/block_(?:0[1-9]|1[0-9])/[^/]+\.edf$'
    rows=[r for r in trees if r['type']=='file' and re.fullmatch(canonical,r['path'])]
    keys={}
    for r in rows:
        match=re.search(r'/session_(\d+)/block_(\d+)/',r['path']);key=tuple(map(int,match.groups()));keys.setdefault(key,[]).append(r)
    missing=[(s,b) for s in range(1,5) for b in range(1,20) if len(keys.get((s,b),[]))!=1]
    if missing:raise RuntimeError(f'Official raw EDF unavailable/ambiguous for {missing}; no subject exclusion; investigate source or NEMAR separately')
    def one(r):
        x=remote.fetch(f'{remote.HF}/datasets/{remote.REPO}/resolve/{revision}/{quote(r["path"])}',dest/'official'/r['path'],r['size'],r.get('lfs',{}).get('oid'))
        print('raw ready',r['path'],flush=True);return x
    with ThreadPoolExecutor(max_workers=4) as pool:download=list(pool.map(one,rows))
    (dest/'raw_download_manifest.json').write_text(json.dumps(download,indent=2)+'\n')
    d=pd.read_parquet(dest/f'official/preprocessed_eeg/{tag(subject)}/experiment_metadata_categories.parquet')
    stimulus=pd.read_parquet(dest/f'official/preprocessed_eeg/{tag(subject)}/stim_order.parquet')
    if len(d)!=len(stimulus):raise RuntimeError('Stimulus metadata length mismatch; no truncation allowed')
    a=np.sort(pd.util.hash_pandas_object(d[stimulus.columns],index=False).to_numpy())
    b=np.sort(pd.util.hash_pandas_object(stimulus,index=False).to_numpy())
    assert np.array_equal(a,b),'Official stimulus/processed metadata contents disagree'
    d=d.reset_index(names='official_index_label');d['metadata_row']=np.arange(len(d));d['image_id']=d.image_path.map(lambda p:int(Path(p).stem))
    spec=importlib.util.spec_from_file_location('aj_recovery_utils',dest/'upstream/preprocessing_utils.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    (dest/'epochs').mkdir(exist_ok=True);records=[];ledgers=[]
    for session in range(1,5):
        for block in range(1,20):
            stem=f'ses-{session:02d}_block-{block:02d}';metaout=dest/'epochs'/f'{stem}.parquet';infoout=dest/'epochs'/f'{stem}.json'
            if metaout.exists() and infoout.exists():
                info=load_json(infoout.read_text())
                if info['official_dropped_disagreements']:
                    raise RuntimeError('Audited boundary erratum cache requires sub08_boundary_repair.py --recover; do not erase its nonzero disagreement count')
                records.append(info);ledgers.append(pd.read_parquet(metaout));continue
            r=keys[(session,block)][0];path=dest/'official'/r['path']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');raw=module.read_eeg_data(str(path.parent))
            meta=d[(d.session==session)&(d.block_id==block)&(d.partition!='oddball')].copy().reset_index(drop=True)
            events,ids=mne.events_from_annotations(raw,regexp='.*stim',verbose=False);reverse={v:k for k,v in ids.items()}
            desc=[reverse[int(v)] for v in events[:,2]];tokens=[(v.split(',')[0],int(v.split(',')[1])) for v in desc]
            positions,method=align(tokens,meta)
            ep=mne.Epochs(raw,events,event_id=ids,tmin=-.2,tmax=1.,baseline=(None,0),preload=True,reject=None,event_repeated='drop',verbose=False)
            if ep.info['sfreq']!=250:ep.resample(250,verbose=False)
            meta['edf_annotation_event_index']=-1;meta['edf_sample_256hz']=-1;meta['annotation_description']='';meta['epoch_index_in_file']=-1
            meta['valid']=False;meta['exclusion_reason']='missing_annotation';meta['epoch_file']=f'epochs/{stem}.npy'
            for i,p in enumerate(positions):
                meta.loc[p,['edf_annotation_event_index','edf_sample_256hz','annotation_description']]=[i,int(events[i,0]),desc[i]]
                meta.loc[p,'exclusion_reason']='|'.join(ep.drop_log[i]) or 'kept'
            keep=positions[ep.selection];meta.loc[keep,'valid']=True;meta.loc[keep,'epoch_index_in_file']=np.arange(len(ep))
            x=ep.get_data().astype(np.float32);assert np.isfinite(x).all() and x.shape[1:]==(32,301)
            mismatch=int(np.count_nonzero(meta.valid.to_numpy()!=~meta.dropped.to_numpy()))
            info={'session':session,'block':block,'edf_file':r['path'],'annotation_alignment':method,'valid_epochs':len(ep),
                  'official_dropped_disagreements':mismatch,'channels':ep.ch_names,'times':ep.times.tolist(),
                  'meas_date':raw.info['meas_date'].isoformat() if raw.info['meas_date'] else None,
                  'recording_duration_s':raw.n_times/raw.info['sfreq']}
            if mismatch:raise RuntimeError(f'{tag(subject)} {stem}: {mismatch} official dropped disagreements; not training this subject until resolved')
            np.save(dest/'epochs'/f'{stem}.npy',x);meta.to_parquet(metaout,index=False);infoout.write_text(json.dumps(info,indent=2)+'\n')
            records.append(info);ledgers.append(meta);print(tag(subject),stem,'verified',len(ep),flush=True)
            del raw,ep,x
    ledger=pd.concat(ledgers,ignore_index=True).sort_values('metadata_row');assert not ledger.metadata_row.duplicated().any()
    assert all(r['channels']==records[0]['channels'] and np.array_equal(r['times'],records[0]['times']) for r in records)
    ledger.to_parquet(dest/'trial_ledger.parquet',index=False)
    d[d.partition=='oddball'].to_parquet(dest/'excluded_oddball.parquet',index=False)
    (dest/'epoch_summary.json').write_text(json.dumps({'subject':tag(subject),'blocks':len(records),'valid':int(ledger.valid.sum()),'official_dropped_disagreements':0,'channels':records[0]['channels'],'times':records[0]['times'],'blocks_detail':records},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,required=True);main(p.parse_args().subject)

