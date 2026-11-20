"""Same audited THINGS training entry points; 32-channel/data-path adaptation."""
from project_paths import load_json, load_json_file, resolve_path, verify_source
from experiments.alljoined.common import *
import argparse,datetime,hashlib,json,subprocess,time
import numpy as np,torch
from omegaconf import OmegaConf
from experiments.leakfree_things.prepare_subject import load_pt

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def event(subject,stage,status,**extra):
    row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'subject':tag(subject),'stage':stage,'status':status,**extra}
    p=ROOT/'status'/f'{tag(subject)}.json';write(p,row)
    with (ROOT/'status/events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def encoder_dir(subject):return ROOT/'checkpoints/encoder'/NAME/f'{tag(subject)}_seed0'
def best_encoder(subject):
    files=list((encoder_dir(subject)/'checkpoints').glob('epoch=*.ckpt'))
    if len(files)!=1:raise RuntimeError(f'Expected exactly one validation-best Encoder, got {files}')
    return files[0].resolve()
def config_path():return REPO/'configs/alljoined/encoder_final.yaml'
def setup():
    config=OmegaConf.load(REPO/'configs/eeg/fixed_fovea_leakfree.yaml')
    config.name=NAME;config.dataset='alljoined';config.models.brain.params.c_num=32
    config.data.data_dir=str(REPO/'data/processed/alljoined/preprocessed');config.data.img_feature_dir=str(REPO/'data/processed/alljoined/encoder_features')
    config.data.image_root=str(REPO/'data/things-eeg/Image_set_Resize');config.save_dir=str(REPO/'checkpoints/alljoined/encoder')
    config.train.weight_decay=1e-4
    config_path().parent.mkdir(parents=True,exist_ok=True)
    if config_path().exists():assert OmegaConf.to_container(OmegaConf.load(config_path()))==OmegaConf.to_container(config)
    else:OmegaConf.save(config,config_path())
    files=['src/encoder/train.py','src/encoder/data.py','src/encoder/models.py','src/encoder/utils.py','src/predictor/train.py','src/predictor/cli.py','src/predictor/data.py','src/predictor/losses.py','src/predictor/model.py']
    current={p:sha(REPO/p) for p in files};path=ROOT/'config/source_hashes.json'
    if path.exists():
        for name,digest in load_json(path.read_text()).items():verify_source(REPO/name,digest)
    else:write(path,current)
def run(subject,stage,command,env):
    marker=ROOT/'status'/f'{tag(subject)}_{stage}_success.json'
    if marker.exists():return
    log=ROOT/'logs'/tag(subject)/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    event(subject,stage,'running',command=command,gpu=env.get('CUDA_VISIBLE_DEVICES'))
    with log.open('a') as f:
        f.write('\n'+json.dumps({'start':time.time(),'command':command})+'\n');f.flush()
        proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=False)
        event(subject,stage,'running',pid=proc.pid,command=command,gpu=env.get('CUDA_VISIBLE_DEVICES'))
        code=proc.wait()
    if code:
        event(subject,stage,'failed',exit_code=code,log=str(log));raise RuntimeError(f'{stage} exit {code}; {log}')
    write(marker,{'success':True,'command':command,'log':str(log),'completed':time.time()})
def attach_preprocessing(path,subject):
    cp=load_pt(path);pre=ROOT/'preprocessing'/tag(subject)
    if 'alljoined_preprocessing' in cp:return
    z=np.load(pre/'mvnn.npz')
    cp['alljoined_preprocessing']={'config':load_json((pre/'preprocessing_config.json').read_text()),
        'validated':load_json((pre/'validated.json').read_text()),
        'matrices':{k:torch.from_numpy(z[k].copy()) for k in z.files},
        'fit_rows':load_json((pre/'mvnn_fit_rows.json').read_text()),
        'trial_indices':str(pre/'trial_indices.parquet'),'trial_indices_sha256':sha(pre/'trial_indices.parquet')}
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(cp,temporary);temporary.replace(path)
def train(subject,gpu):
    setup();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO/'src')+os.pathsep+str(REPO),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    run(subject,'recover',[PYTHON,str(CODE_ROOT/'recover.py'),'--subject',str(subject)],env)
    run(subject,'prepare',[PYTHON,str(CODE_ROOT/'prepare.py'),'--subject',str(subject)],env)
    run(subject,'data_checks',[PYTHON,str(CODE_ROOT/'check_data.py'),'--subject',str(subject)],env)
    assert load_json((ROOT/'preprocessing'/tag(subject)/'validated.json').read_text())['passed']
    wait_for_gpu_idle(gpu)
    enc=[PYTHON,'-m','encoder.train','--config',str(config_path()),'--dataset','alljoined','--subjects',tag(subject),
         '--seed','0','--exp_setting','intra-subject','--brain_backbone','EEGProjectLayer','--vision_backbone','RN50','--epoch','50','--lr','1e-4','--skip-test']
    run(subject,'encoder',enc,env)
    for p in (encoder_dir(subject)/'checkpoints').glob('*.ckpt'):attach_preprocessing(p,subject)
    best=best_encoder(subject);cp=load_pt(best)
    assert cp['state_dict']['brain.model.0.weight'].shape==(1024,8000)
    assert any('val_top1_acc' in k for k in cp['callbacks'])
    write(ROOT/'selection'/f'{tag(subject)}_encoder.json',{'checkpoint':str(best),'sha256':sha(best),'epoch':cp['epoch'],'callbacks':{k:str(v) for k,v in cp['callbacks'].items()},'rule':'best validation Top-1, fixed 50 epochs; test unread'})
    del cp
    common=[PYTHON,'-m','predictor.train','--sub',str(subject),'--gpu','0','--seed','2023','--batch_size','64','--weight_decay','1e-5',
        '--n_layers','4','--generator_arch','full','--clip_input','fovea','--semantic_mode','ubp_margin','--lambda_sem','0','--lambda_time','1','--lambda_corr','0.8',
        '--lambda_ubp','0.5','--lambda_eeg_nce','0.1','--lambda_margin','0','--lambda_div','0',
        '--data_dir',str(ROOT/'preprocessed'),'--image_root',str(REPO/'data/things-eeg/Image_set_Resize'),
        '--feature_dir',str(ROOT/'generator_features'),'--ubp_ckpt',str(best),'--skip-test']
    for stage in ['stage1','stage2']:
        wait_for_gpu_idle(gpu)
        ckptroot=ROOT/'checkpoints/generator'/stage
        command=common+['--ckpt_dir',str(ckptroot),'--result_dir',str(ROOT/'generator'/stage)]
        if stage=='stage1':command+=['--epochs','100','--early_stop_patience','15','--early_stop_metric','loss','--lr','1e-4','--lambda_freq','0.2','--lambda_band','0.2']
        else:command+=['--epochs','25','--early_stop_patience','8','--early_stop_metric','semantic_guard','--lr','1.5e-5','--lambda_freq','0','--lambda_band','0',
                        '--lambda_band_corr','0.32','--band_weights','delta=2.0,beta=1.5,gamma=1.0','--gamma_fmax','45',
                        '--init_ckpt',str(ROOT/'checkpoints/generator/stage1'/tag(subject)/'last.pt')]
        run(subject,stage,command,env)
        for name in ['best.pt','last.pt']:attach_preprocessing(ckptroot/tag(subject)/name,subject)
        cp=load_pt(ckptroot/tag(subject)/'best.pt');a=cp['args']
        assert cp['n_channels']==32 and cp['seq_len']==250 and a['skip_test']
        assert Path(resolve_path(a['ubp_ckpt']))==best
        assert np.isfinite(cp['best_val'])
        write(ROOT/'selection'/f'{tag(subject)}_{stage}.json',{'checkpoint':str(ckptroot/tag(subject)/'best.pt'),'sha256':sha(ckptroot/tag(subject)/'best.pt'),
            'epoch':cp['epoch'],'best_val':cp['best_val'],'args':a,'selection_used_test':False})
        del cp
    event(subject,'selected','training_complete',test_used_for_selection=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    try:train(args.subject,args.gpu)
    except Exception as exc:event(args.subject,'pipeline','failed',error=repr(exc));raise

