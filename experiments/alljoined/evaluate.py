"""Validation diagnostics and one-shot final test after frozen model selection."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,json
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from experiments.leakfree_things.prepare_subject import load_pt
from predictor.data import load_frozen_ubp_brain,load_clip_features
from predictor.model import load_any_generator_from_checkpoint
from figures.eeg_gen_eval.compute.metrics import global_pearson,global_rmse,per_channel_pearson,psd_mean_sem,bandpower_metrics
from experiments.alljoined.train_subject import best_encoder,sha,write,setup

def corr(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else float('nan')
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def retrieval(eeg,features,brain,device,batch):
    with torch.no_grad():z=F.normalize(brain(eeg.to(device)),dim=1).cpu()
    features=F.normalize(features,dim=1);one=five=n=0
    for start in range(0,len(eeg),batch):
        sim=z[start:start+batch]@features[start:start+batch].T
        order=sim.topk(5,dim=-1).indices.numpy();labels=np.arange(len(order))
        one+=int((order[:,0]==labels).sum());five+=int((order==labels[:,None]).any(1).sum());n+=len(order)
    return {'top1':one/n,'top5':five/n,'images':n,'retrieval_candidates':f'batches of {batch}; final batch may be smaller'}
def training_quality(subject):
    marker=ROOT/'quality'/f'{tag(subject)}.json'
    if marker.exists():return load_json(marker.read_text())
    rows=pd.read_parquet(ROOT/'preprocessing'/tag(subject)/'trial_indices.parquet');rows=rows[rows.role=='train']
    source=cache(subject);arrays={p:np.load(source/p,mmap_mode='r') for p in rows.epoch_file.unique()}
    n=0;sa=sb=saa=sbb=sab=0.;eligible=0
    for _,g in rows.groupby('image_id',sort=True):
        sessions=sorted(g.session.unique())
        if len(sessions)<2:continue
        x=np.stack([arrays[r.epoch_file][int(r.epoch_index_in_file),:,50:300] for r in g.itertuples()]).astype(np.float64)
        # Separate acquisition sessions, rather than random neighboring RSVP
        # repetitions, keep the two quality halves in distinct recordings.
        first=g.session.to_numpy()==sessions[0]
        a=x[first].mean(0);b=x[~first].mean(0)
        n+=a.size;sa+=a.sum();sb+=b.sum();saa+=(a*a).sum();sbb+=(b*b).sum();sab+=(a*b).sum();eligible+=1
    r=float((sab-sa*sb/n)/np.sqrt((saa-sa*sa/n)*(sbb-sb*sb/n))) if n else float('nan')
    result=clean({'training_split_half_r':r,'spearman_brown':2*r/(1+r) if r>-1 else None,'eligible_training_images':eligible,
        'source':'retained training trials only, unwhitened baseline-corrected original electrodes, 0..1s; lower acquisition session vs other session(s), pooled cross-session repeatability correlation',
        'quality_halves_share_recording_sessions':False,
        'limitation':'Descriptive cross-session repeatability proxy, may include common ERP and session drift; not a noise ceiling or causal explanation. Spearman-Brown is secondary and assumes parallel halves, not established here. Never used for subject exclusion'})
    write(marker,result);return result

def evaluate(subject,gpu,pilot=False):
    setup();dest=ROOT/'evaluation'/tag(subject);dest.mkdir(parents=True,exist_ok=True)
    selected=best_encoder(subject);generator=ROOT/'checkpoints/generator/stage2'/tag(subject)/'best.pt'
    hashes={'encoder_sha256':sha(selected),'generator_sha256':sha(generator)}
    if (dest/'metrics.json').exists():
        assert load_json((dest/'test_claim.json').read_text())==hashes
        print('one-shot test already complete',tag(subject),flush=True);return
    wait_for_gpu_idle(int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]))
    torch.cuda.set_device(0);device=torch.device('cuda:0')
    brain=load_frozen_ubp_brain({'target':'encoder.models.EEGProjectLayer','params':{'c_num':32,'z_dim':1024,'timesteps':[0,250]}},str(selected),device)
    if brain is None:raise RuntimeError('Required new Encoder missing; semantic loss/evaluation must not silently disappear')
    validation=load_pt(ROOT/'preprocessed'/tag(subject)/'val.pt')
    vy=torch.from_numpy(validation['eeg']).float().mean(1)
    feature=load_pt(ROOT/'encoder_features/FoveaBlur'/f'{NAME}_val.pt')['img_features']
    vi=torch.stack([feature[str(p)] for p in validation['img'][:,0]])
    val=retrieval(vy,vi,brain,device,200)
    cp=load_pt(selected);expected=[v['best_model_score'] for k,v in cp['callbacks'].items() if k.startswith('ModelCheckpoint')][0]
    assert abs(val['top1']-float(expected))<1e-6,'Validation retrieval differs from checkpoint selection score'
    quality=training_quality(subject)
    write(dest/'validation_diagnostics.json',{'encoder':val,'quality':quality,'test_unread_so_far':True})
    # Exclusive marker prevents silently repeating a test after changing config.
    claim=dest/'test_claim.json'
    with claim.open('x') as f:json.dump(hashes,f,indent=2)
    p=load_pt(ROOT/'preprocessed'/tag(subject)/'test.pt');y=np.asarray(p['eeg'],dtype=np.float32).mean(1)
    assert y.shape==(200,32,250)
    modelcp=load_pt(generator);assert modelcp['args']['skip_test'] and modelcp['n_channels']==32
    assert Path(resolve_path(modelcp['args']['ubp_ckpt']))==selected
    model=load_any_generator_from_checkpoint(modelcp,32,250).to(device).eval()
    clip=load_clip_features(np.asarray(p['img'])[:,0],data_dir=str(ROOT/'preprocessed'),mode='test',clip_input='fovea',device=device,
        image_root=str(REPO/'data/things-eeg/Image_set_Resize'),feature_dir=str(ROOT/'generator_features'))
    with torch.no_grad():
        pred=torch.cat([model(clip[i:i+100].to(device)).cpu() for i in range(0,200,100)]).numpy()
        eg=F.normalize(brain(torch.from_numpy(pred).to(device)),dim=1).cpu().numpy()
        et=F.normalize(brain(torch.from_numpy(y).to(device)),dim=1).cpu().numpy()
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite model output, technical failure')
    encoder_features=load_pt(ROOT/'encoder_features/FoveaBlur'/f'{NAME}_test.pt')['img_features']
    encimage=torch.stack([encoder_features[str(q)] for q in p['img'][:,0]])
    enc=retrieval(torch.from_numpy(y),encimage,brain,device,200)
    channels=list(p['ch_names']);roi=[i for i,c in enumerate(channels) if c.startswith(('O','PO'))]
    pc=per_channel_pearson(pred,y);tp=psd_mean_sem(y);pp=psd_mean_sem(pred)
    positive=np.isfinite(tp['psd_mean'])&np.isfinite(pp['psd_mean'])&(tp['psd_mean']>0)&(pp['psd_mean']>0)
    select=(tp['freqs_hz']>=.5)&(tp['freqs_hz']<=45)&positive
    full=(tp['freqs_hz']>=.5)&positive
    bp=bandpower_metrics(pred,y)
    rsa_encoder=float(spearmanr(pdist(eg,'cosine'),pdist(et,'cosine')).statistic)
    rdm_true=pdist(y.reshape(200,-1),'correlation');rdm_pred=pdist(pred.reshape(200,-1),'correlation')
    rsa_eeg=float(spearmanr(rdm_true,rdm_pred).statistic)
    metrics={'subject':tag(subject),'encoder_top1':enc['top1'],'encoder_top5':enc['top5'],
        'validation_encoder_top1':val['top1'],'validation_encoder_top5':val['top5'],
        'rmse':global_rmse(pred,y),'mae':float(np.abs(pred-y).mean()),'pooled_pearson':global_pearson(pred,y),
        'mean_channel_pearson':float(pc.mean()),'occipital_ERP_pearson':corr(pred[:,roi].mean((0,1)),y[:,roi].mean((0,1))),
        'whole_scalp_log_PSD_pearson':corr(np.log10(pp['psd_mean'][select]),np.log10(tp['psd_mean'][select])),
        'whole_scalp_log_PSD_pearson_full_Nyquist':corr(np.log10(pp['psd_mean'][full]),np.log10(tp['psd_mean'][full])),
        'PSD_definition':'paper primary: channel/sample-mean Welch PSD, log10, positive finite bins 0.5..45 Hz; supplementary full spectrum 0.5..Nyquist',
        'mean_bandpower_pearson':float(np.mean(list(bp['bandpower_correlation'].values()))),
        'pred_clip_cosine':float((eg*F.normalize(clip,dim=1).numpy()).sum(1).mean()),
        'gt_pred_encoder_cosine':float((eg*et).sum(1).mean()),'encoder_RSA_spearman':rsa_encoder,'EEG_RSA_spearman':rsa_eeg,
        'training_split_half_r':quality['training_split_half_r'],'training_split_half_spearman_brown':quality['spearman_brown'],
        'quality_eligible_images':quality['eligible_training_images'], 'all_subjects_included_regardless_of_performance':True,
        'signal_space':'session-specific training-only MVNN space, true retained-repetition means; not raw electrode predictions',
        'ROI_space_limit':'Occipital-labelled rows are evaluated in MVNN space. Spatial whitening mixes electrodes; this is not an unwhitened anatomical ROI result.',
        'test_repeats_min':int(np.min(p['valid_repeats'])),'test_repeats_max':int(np.max(p['valid_repeats'])),
        'semantic_limit':'Encoder cosine/RSA reuse the training-loss Encoder; direct EEG RSA is encoder-independent representational evidence, not proof of semantic causality',
        **hashes}
    np.save(dest/'y_true.npy',y);np.save(dest/'y_pred.npy',pred);np.save(dest/'image_ids.npy',p['image_ids'])
    np.save(dest/'per_channel_pearson.npy',pc);np.save(dest/'EEG_RDM_true.npy',squareform(rdm_true));np.save(dest/'EEG_RDM_pred.npy',squareform(rdm_pred))
    np.savez(dest/'spectral.npz',freqs=tp['freqs_hz'],psd_true=tp['psd_mean'],psd_pred=pp['psd_mean'],bandpower_true=bp['bandpower_true'],bandpower_pred=bp['bandpower_pred'])
    np.savez(dest/'erp.npz',true=y.mean(0),pred=pred.mean(0),channels=np.asarray(channels),times=np.asarray(p['times']))
    write(dest/'metrics.json',clean(metrics))
    # Pipeline gates are structural/numerical only, never minimum test accuracy.
    assert np.isfinite(metrics['rmse']) and np.isfinite(metrics['pooled_pearson']) and np.isfinite(metrics['mean_channel_pearson'])
    write(dest/'workflow_passed.json',{'passed':True,'test_accuracy_threshold_used':False,'pilot':pilot,'selection_frozen_before_test':True,**hashes})
    print(json.dumps(clean(metrics),indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--pilot',action='store_true');a=p.parse_args();evaluate(a.subject,a.gpu,a.pilot)

