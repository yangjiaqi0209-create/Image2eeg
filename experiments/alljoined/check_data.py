"""Independent, mandatory pretraining input/order/identity check."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,hashlib,json
import numpy as np,pandas as pd
from experiments.leakfree_things.prepare_subject import load_pt

def check(subject):
    pre=ROOT/'preprocessing'/tag(subject);d=pd.read_parquet(pre/'trial_indices.parquet')
    official=np.load(REPO/'data/processed/things_eeg2/split/image_split.npz')
    canonical=pd.read_parquet(AUDIT/'recovery_sub01/image_identity_checks.parquet').set_index('image_id').model_image_path
    mapped=d.image_id.map(canonical)
    assert mapped.notna().all() and np.array_equal(mapped.astype(str).to_numpy(),d.model_image_path.astype(str).to_numpy()),'Image ID/category/path mismatch'
    checks={};sets={}
    for role in ['train','val','test']:
        p=load_pt(ROOT/'preprocessed'/tag(subject)/f'{role}.pt')
        names=np.asarray(p['img'])[:,0].astype(str);available=set(names)
        expected=np.asarray([n for n in official[role+'_images'].astype(str) if n in available])
        assert np.array_equal(names,expected),'Formal frozen image order changed'
        if role in ['val','test']:assert np.array_equal(names,official[role+'_images'].astype(str))
        assert p['eeg'].shape==(len(names),1,32,250) and np.isfinite(p['eeg']).all()
        assert not np.any(np.all(p['eeg']==0,axis=(1,2,3)))
        assert len(set(names))==len(names)
        counts=d[d.role==role].groupby('model_image_path').size().reindex(names).to_numpy()
        assert np.array_equal(counts,p['valid_repeats'])
        sets[role]=available
        checks[role]={'images':len(names),'retained_trials':int(counts.sum()),'formal_order_matches':True,'no_zero_padding':True}
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    assert d.loc[d.role.isin(['train','val','test']),'valid'].all()
    fit=load_json((pre/'mvnn_fit_rows.json').read_text());ids={v for f in fit for v in f['metadata_rows']}
    assert ids==set(d.loc[d.role=='train','metadata_row'])
    assert not ids&set(d.loc[d.role.isin(['val','test']),'metadata_row'])
    summary=load_json((pre/'validated.json').read_text())
    assert all(v==0 for k,v in summary['overlap'].items() if 'shared_256hz_samples' in k)
    if summary['official_dropped_disagreements']:
        from experiments.alljoined.sub08_boundary_repair import validate_recovery_summary
        source=cache(subject)
        proof=validate_recovery_summary(source,load_json((source/'epoch_summary.json').read_text()))
        assert subject==8 and proof and summary['unresolved_official_dropped_disagreements']==0
        expected_rows=set(proof['metadata_rows'])
        excluded=d[d.metadata_row.isin(expected_rows)]
        assert len(excluded)==len(expected_rows) and not excluded.valid.any() and (excluded.role=='invalid').all()
        assert not ids.intersection(expected_rows)
        checks['verified_incomplete_boundary_epochs_excluded']=len(expected_rows)
    checks.update(passed=True,canonical_stimulus_mapping_verified=True,training_only_mvnn_verified=True,
                  trial_indices_sha256=hashlib.sha256((pre/'trial_indices.parquet').read_bytes()).hexdigest())
    (pre/'independent_data_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,required=True);check(p.parse_args().subject)
