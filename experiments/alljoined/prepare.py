"""Approved image split + train/validation buffer. No training or test inference."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,gc,hashlib,importlib.util,json
import numpy as np,pandas as pd,scipy.linalg,torch
from threadpoolctl import threadpool_limits
from experiments.leakfree_things.prepare_subject import load_pt

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')

def prepare(subject):
    source=cache(subject);out=ROOT/'preprocessing'/tag(subject);out.mkdir(parents=True,exist_ok=True)
    ready=out/'validated.json'
    if ready.exists():
        assert all((ROOT/'preprocessed'/tag(subject)/f'{s}.pt').exists() for s in ['train','val','test'])
        print('validated data already exist',tag(subject),flush=True);return
    d=pd.read_parquet(source/'trial_ledger.parquet').copy()
    summary=load_json((source/'epoch_summary.json').read_text())
    from experiments.alljoined.sub08_boundary_repair import validate_recovery_summary
    boundary_resolution=validate_recovery_summary(source,summary)
    from preprocess.convert_alljoined import _build_image_lookup
    lookup=_build_image_lookup(REPO/'data/things-eeg/Image_set_Resize')
    d['model_image_path']=[lookup[(r.partition,int(r.category_num),int(r.category_img_num))] for r in d.itertuples()]
    images=d[d.partition=='stim_train'].drop_duplicates('image_id').sort_values(['category_num','category_img_num'])
    frozen=np.load(REPO/'data/processed/things_eeg2/split/image_split.npz')
    paths=images.model_image_path.astype(str).to_numpy()
    assert len(paths)==16540
    assert np.array_equal(paths[frozen['train_index']],frozen['train_images'].astype(str))
    assert np.array_equal(paths[frozen['val_index']],frozen['val_images'].astype(str))
    vp=set(frozen['val_images'].astype(str))
    d['role']=np.where(d.partition=='stim_test','test',np.where(d.model_image_path.isin(vp),'val','train'))
    d.loc[~d.valid,'role']='invalid'
    d['buffer_excluded']=False
    for _,g in d[d.valid].groupby('epoch_file'):
        v=np.sort(g.loc[g.role=='val','edf_sample_256hz'].to_numpy(dtype=int))
        tr=g[g.role=='train'];x=tr.edf_sample_256hz.to_numpy(dtype=int)
        if len(v):
            mask=np.searchsorted(v-51,x+257,side='left')>np.searchsorted(v+257,x-51,side='right')
            idx=tr.index[mask];d.loc[idx,'role']='buffer_unused';d.loc[idx,'buffer_excluded']=True
    sys.path.insert(0,str(AUDIT/'rsvp_split_review'))
    from compare_splits import overlap_metrics
    overlaps=overlap_metrics(d,d.role)
    assert all(v==0 for k,v in overlaps.items() if 'shared_256hz_samples' in k)
    assert not d.loc[d.role.isin(['train','val','test'])].duplicated(['epoch_file','epoch_index_in_file']).any()
    d.to_parquet(out/'trial_indices.parquet',index=False)
    repeats=d[d.role.isin(['train','val','test'])].groupby(['role','image_id','model_image_path']).size().rename('valid_repeats').reset_index()
    repeats.to_csv(out/'valid_repeats.csv',index=False)
    module_path=source/'upstream/preprocessing_utils.py'
    spec=importlib.util.spec_from_file_location('alljoined_official_pinned',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    ws=[];inverses=[];covs=[];fit=[]
    for session in range(1,5):
        selected=d[(d.role=='train')&(d.session==session)]
        arrays={p:np.load(source/p,mmap_mode='r') for p in selected.epoch_file.unique()}
        cc=[]
        for _,g in selected.groupby('image_id',sort=True):
            x=np.stack([arrays[r.epoch_file][int(r.epoch_index_in_file)] for r in g.itertuples()]).astype(np.float64)
            cc.append(module._compute_sigma_cond('epochs',x))
        cov=np.mean(cc,axis=0);eigen=np.linalg.eigvalsh(cov)
        if eigen[0]<=0:raise RuntimeError('Nonpositive covariance; no hyperparameter adjustment allowed')
        w=scipy.linalg.fractional_matrix_power(cov,-.5);inv=np.linalg.inv(w)
        assert np.max(np.abs(w@cov@w.T-np.eye(32)))<1e-8
        ws.append(w);inverses.append(inv);covs.append(cov)
        fit.append({'session':session,'metadata_rows':selected.metadata_row.tolist(),'conditions':len(cc),'trials':len(selected),'condition_number':float(eigen[-1]/eigen[0])})
        print(tag(subject),'train-only MVNN',session,'trials',len(selected),flush=True)
    np.savez(out/'mvnn.npz',whitening=np.asarray(ws),inverse=np.asarray(inverses),covariance=np.asarray(covs))
    dump(out/'mvnn_fit_rows.json',fit)
    assert {i for s in fit for i in s['metadata_rows']}==set(d.loc[d.role=='train','metadata_row'])
    dest=ROOT/'preprocessed'/tag(subject);dest.mkdir(parents=True,exist_ok=True)
    statistics={};inverse_errors=[]
    for role in ['train','val','test']:
        selected=d[d.role==role]
        available=set(selected.image_id)
        path_to_id=d.drop_duplicates('model_image_path').set_index('model_image_path').image_id.to_dict()
        formal_paths=frozen[{'train':'train_images','val':'val_images','test':'test_images'}[role]].astype(str)
        ids=[int(path_to_id[p]) for p in formal_paths if path_to_id[p] in available]
        assert len(ids)==len(available)
        order={v:i for i,v in enumerate(ids)}
        count=np.zeros(len(ids),dtype=int);sums=np.zeros((len(ids),32,250),dtype=np.float64)
        rawsum=np.zeros_like(sums)
        for name,g in selected.groupby('epoch_file',sort=True):
            raw=np.load(source/name,mmap_mode='r')[g.epoch_index_in_file.to_numpy(dtype=int),:,50:300].astype(np.float64)
            s=int(g.session.iloc[0])-1;white=np.matmul(ws[s],raw).astype(np.float32)
            recovered=np.matmul(inverses[s],white.astype(np.float64))
            inverse_errors.append(float(np.linalg.norm(recovered-raw)/np.linalg.norm(raw)))
            idx=np.asarray([order[v] for v in g.image_id]);np.add.at(sums,idx,white);np.add.at(rawsum,idx,raw);np.add.at(count,idx,1)
        means=(sums/count[:,None,None]).astype(np.float32)
        assert count.min()>0 and np.isfinite(means).all() and not np.any(np.all(means==0,axis=(1,2)))
        meta=selected.drop_duplicates('image_id').set_index('image_id').loc[ids]
        payload={'eeg':means[:,None],'img':meta.model_image_path.to_numpy()[:,None],
                 'label':meta.category_num.to_numpy()[:,None],'text':meta.category_name.to_numpy()[:,None],
                 'session':np.zeros((len(ids),1),dtype=np.int64),'ch_names':summary['channels'],
                 'times':np.asarray(summary['times'])[50:300],'trial_format':'per_image',
                 'valid_repeats':count,'image_ids':np.asarray(ids),
                 'preprocessing_manifest':str(out/'validated.json')}
        torch.save(payload,dest/f'{role}.pt',pickle_protocol=5)
        np.save(dest/f'{role}_raw_electrode_mean_V.npy',(rawsum/count[:,None,None]).astype(np.float32))
        statistics[role]={'images':len(ids),'trials':len(selected),'concepts':int(meta.category_name.nunique()),'min_repeats':int(count.min()),'max_repeats':int(count.max())}
    assert statistics['val']['images']==740 and statistics['test']['images']==200
    if subject==1:assert statistics['train']['trials']==42528 and statistics['train']['images']==15593
    assert max(inverse_errors)<1e-5
    # Reuse only frozen visual CLIP features, never EEG/brain model weights.
    for kind in ['encoder','generator']:
        old=REPO/'results/things_eeg2_leakfree'/f'{kind}_features'
        if kind=='encoder':
            target=ROOT/'encoder_features/FoveaBlur';target.mkdir(parents=True,exist_ok=True)
            for role in ['train','val','test']:
                src=old/'FoveaBlur'/f'THINGSEEG2_leakfree_{role}.pt'
                cp=load_pt(src);needed=set(d.loc[d.role==role,'model_image_path'])
                assert needed<=set(cp['img_features']);del cp
                link=target/f'{NAME}_{role}.pt'
                if not link.exists():link.symlink_to(src.resolve())
        else:
            target=ROOT/'generator_features';target.mkdir(exist_ok=True)
            for src in old.glob('*.pt'):
                link=target/src.name
                if not link.exists():link.symlink_to(src.resolve())
    config={'subject':tag(subject),'seed_image_split':2023,'encoder_seed':0,'generator_seed':2023,
            'split_rule':'exact formal THINGS image split; user-approved removal of only train/val raw-overlapping training epochs',
            'no_train_train_purge':True,'buffer_original_samples':[-51,257],'source_cache':str(source),
            'mvnn':'official epochs-mode, per session, train rows only','mean':'all retained valid repetitions; no zero padding',
            'zscore':None,'baseline':[None,0],'sfreq':250,'channels':summary['channels'],'n_timepoints':250,
            'dtype':'float32 valid-repetition means; no storage padding','cross_session_mean_single_inverse':False}
    if boundary_resolution:
        config['boundary_resolution']=boundary_resolution
    dump(out/'preprocessing_config.json',config)
    dump(ready,{'passed':True,'counts':statistics,'overlap':overlaps,'formal_image_order_preserved':True,'buffer_training_trials_removed':int(d.buffer_excluded.sum()),
                'max_inverse_relative_error':max(inverse_errors),'official_dropped_disagreements':int(summary['official_dropped_disagreements']),
                'resolved_official_dropped_disagreements':int(summary.get('resolved_official_dropped_disagreements',0)),
                'unresolved_official_dropped_disagreements':int(summary.get('unresolved_official_dropped_disagreements',0)),
                'boundary_resolution':boundary_resolution,'test_used_for_selection':False,
                'metadata_indices_sha256':hashlib.sha256((out/'trial_indices.parquet').read_bytes()).hexdigest()})
    print(tag(subject),'DATA PASSED',statistics,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--subject',type=int,required=True);args=ap.parse_args()
    with threadpool_limits(limits=1):prepare(args.subject)

