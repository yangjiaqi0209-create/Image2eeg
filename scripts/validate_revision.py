"""Read-only migration acceptance: imports, split independence, strict weights and CPU inference.

Run from root with bash scripts/python.sh scripts/validate_revision.py.
All outputs go to the explicitly selected new report file; no official result is written.
"""
from pathlib import Path
import argparse,gc,importlib,json,traceback
import numpy as np
import torch
from project_paths import REPO,load_json,resolve_path,sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--inference-only',action='store_true',help='Reuse the separately retained strict-weight report')
    args=parser.parse_args();assert not args.report.exists(),args.report
    torch.set_num_threads(2)
    report={'device':'cpu','cuda_available':torch.cuda.is_available(),'checks':[]}
    def check(name,func):
        try:
            detail=func();row={'name':name,'status':'passed','detail':detail}
        except Exception:
            row={'name':name,'status':'failed','traceback':traceback.format_exc()}
        report['checks'].append(row);print(name,row['status'],flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,default=str))
    modules=['encoder.models','encoder.data','encoder.train','predictor.model','predictor.train',
             'preprocess.process_eeg_whiten','preprocess.convert_alljoined','preprocess.unified_mvnn',
             'experiments.leakfree_things.prepare_subject','experiments.leakfree_things.evaluate',
             'experiments.alljoined.train_subject','experiments.alljoined.evaluate',
             'experiments.external_baselines.experiment','experiments.independent_semantics.decoder',
             'figures.eeg_gen_eval.plots.plot_final_fig1','figures.eeg_gen_eval.plots.plot_final_fig2']
    for module in modules:check('import '+module,lambda m=module:importlib.import_module(m).__file__)
    def split_check():
        z=np.load(REPO/'data/processed/things_eeg2/split/image_split.npz')
        for a,b in [('train_images','val_images'),('train_images','test_images'),('val_images','test_images')]:
            assert not set(z[a])&set(z[b])
        counts={mode:len(z[mode+'_images']) for mode in ['train','val','test']}
        assert counts=={'train':15800,'val':740,'test':200};return counts
    check('THINGS frozen split independence',split_check)
    from predictor.model import load_any_generator_from_checkpoint
    from predictor.data import load_frozen_ubp_brain
    from experiments.leakfree_things.prepare_subject import load_pt
    for dataset,n_subjects,channels in [('things_eeg2',10,63),('alljoined',20,32)]:
        result_root=REPO/'results'/('things_eeg2_leakfree' if dataset=='things_eeg2' else 'alljoined')
        for sub in ([] if args.inference_only else range(1,n_subjects+1)):
            tag=f'sub-{sub:02d}'
            def weights(dataset=dataset,channels=channels,result_root=result_root,tag=tag,sub=sub):
                record=load_json((result_root/'selection'/f'{tag}_encoder.json').read_text())
                encoder=resolve_path(record['checkpoint']);assert encoder.is_file(),encoder
                if 'sha256' in record:assert sha256(encoder)==record['sha256']
                brain=load_frozen_ubp_brain({'target':'encoder.models.EEGProjectLayer',
                    'params':{'c_num':channels,'z_dim':1024,'timesteps':[0,250]}},str(encoder),torch.device('cpu'))
                assert brain is not None and not any(p.requires_grad for p in brain.parameters())
                loaded=[]
                for stage in ['stage1','stage2']:
                    path=REPO/'checkpoints'/dataset/'generator'/stage/tag/'best.pt'
                    cp=load_pt(path);a=cp['args'];assert a['skip_test']
                    assert resolve_path(a['ubp_ckpt'])==encoder,(a['ubp_ckpt'],encoder)
                    assert cp['n_channels']==channels and cp['seq_len']==250
                    model=load_any_generator_from_checkpoint(cp,channels,250).eval()
                    loaded.append({'stage':stage,'epoch':cp['epoch'],'sha256':sha256(path)})
                    del cp,model
                del brain;gc.collect()
                return {'encoder':str(encoder.relative_to(REPO)),'generator':loaded}
            check(f'strict weights {dataset} {tag}',weights)
        def inference(dataset=dataset,channels=channels,result_root=result_root):
            cache=REPO/'data/processed'/dataset/'generator_features/generator_fovea_medium_k51_RN50_test.pt'
            cp=load_pt(REPO/'checkpoints'/dataset/'generator/stage2/sub-01/best.pt')
            model=load_any_generator_from_checkpoint(cp,channels,250).eval()
            feature=load_pt(cache)['img_features']
            if dataset=='things_eeg2':
                names=np.load(REPO/'data/processed/things_eeg2/split/image_split.npz')['test_images'].astype(str)
                truth=np.load(result_root/'revision_experiments/fig1_leakfree_cache/sub-01/y_true.npy')
                saved=load_json((result_root/'evaluation/sub-01_final.json').read_text())
            else:
                payload=load_pt(REPO/'data/processed/alljoined/preprocessed/sub-01/test.pt')
                names=np.asarray(payload['img'])[:,0].astype(str)
                del payload
                truth=np.load(result_root/'evaluation/sub-01/y_true.npy')
                saved=load_json((result_root/'evaluation/sub-01/metrics.json').read_text())
            x=torch.stack([torch.as_tensor(feature[str(n)]) for n in names]).float()
            with torch.inference_mode():
                prediction=torch.cat([model(x[i:i+100]) for i in range(0,200,100)]).numpy()
            assert prediction.shape==(200,channels,250) and np.isfinite(prediction).all()
            from figures.eeg_gen_eval.compute.metrics import global_pearson,mean_per_channel_pearson
            pooled=float(global_pearson(prediction,truth));channel=float(mean_per_channel_pearson(prediction,truth))
            # Save field names first; compare only the already recorded metric, never fill a missing result.
            expected=saved.get('pooled_pearson',saved.get('pooled_waveform_r'))
            if expected is None and 'primary_new_encoder_new_whitening' in saved:
                expected=saved['primary_new_encoder_new_whitening']['pooled_pearson']
            if expected is None and isinstance(saved.get('generator'),dict):expected=saved['generator'].get('pooled_pearson')
            if expected is None:raise KeyError('Saved pooled Pearson absent: '+str(saved.keys()))
            assert abs(pooled-float(expected))<1e-4,(pooled,expected)
            return {'n_images':200,'pooled_pearson':pooled,'mean_channel_pearson':channel,
                    'saved_pooled_pearson':expected,'tolerance':1e-4}
        check('CPU inference '+dataset+' sub-01',inference)
    failures=[r['name'] for r in report['checks'] if r['status']=='failed']
    print('Failures:',failures,flush=True)
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
