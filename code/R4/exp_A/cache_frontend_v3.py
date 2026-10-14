"""New actual TRAIN cache: linear mel-power for fold gain, peak controls, running raw sides."""
from concurrent.futures import ProcessPoolExecutor
import os,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch,soundfile as sf
from frontend_v3 import Frontend,FrontendConfig,RunningSide,resample,configs,LEVEL_COLUMNS,BAND_COLUMNS
from audit_frontend_v3 import ROOT,STAGE,note
def decode(row):
    i,file,path=row;w,sr=sf.read(path,dtype='float32');w=w.mean(1) if w.ndim==2 else w
    raw=np.asarray(w,np.float32);waves={}
    for rate in [16000,48000]:
        x=resample(raw,sr,rate);window=rate//50
        if len(x)<window:x=np.pad(x,(0,window-len(x)))
        waves[rate]=x
    stats=RunningSide();stats.update_samples(raw)
    return i,file,waves,stats,sr
def main():
    torch.set_num_threads(2);d=pd.read_csv(STAGE/'shared/data_index.csv');d=d[d.db.isin(['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE'])].sort_values('file').reset_index(drop=True)
    folder=ROOT/'cache_v3';folder.mkdir(exist_ok=True)
    if (folder/'index.csv').exists():assert pd.read_csv(folder/'index.csv').file.tolist()==d.file.tolist()
    else:d[['file','path']].to_csv(folder/'index.csv',index=False)
    (ROOT/'A1_candidates_v3.json').write_text(json.dumps({'candidates':configs(),'fits':360,'seeds':[0,1,2],'dimensions':5},indent=2))
    missing=[(i,r.file,r.path) for i,r in enumerate(d.itertuples()) if not (folder/f'{i:05d}.npz').exists()]
    front={(sr,mel):Frontend(FrontendConfig(sr,mel),device='cuda') for sr,mel in [(16000,48),(48000,48),(48000,64)]}
    note(f'新A1 cache实际PID{os.getpid()}，缺{len(missing)}/11020，原幅度线性mel power保存，按fold gain后再floor/log；运行原幅度侧信息独立。')
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for start in range(0,len(missing),16):
            batch=list(pool.map(decode,missing[start:start+16]));features=[{} for _ in batch]
            for sr in [16000,48000]:
                lengths=[len(row[2][sr]) for row in batch];xx=np.zeros((len(batch),max(lengths)),np.float32)
                for j,row in enumerate(batch):xx[j,:lengths[j]]=row[2][sr]
                x=torch.as_tensor(xx,device='cuda');f=front[(sr,48)];frames=x.unfold(1,f.n_fft,f.hop)
                with torch.inference_mode():
                    power=torch.fft.rfft(frames*f.window,dim=-1).abs().square();peak=x.abs().amax(1).clamp_min(1e-30)
                    if sr==48000:
                        p_cpu=power.cpu().numpy().astype(np.float64);rms_power=frames.double().square().mean(-1).cpu().numpy()
                        for j,row in enumerate(batch):
                            n=(lengths[j]-f.n_fft)//f.hop+1;stats=row[3];active=rms_power[j,:n]>1e-5;stats.active_power=float(rms_power[j,:n][active].sum());stats.active_frames=int(active.sum());stats.update_band_power(p_cpu[j,:n]);features[j]['side']=stats.value();features[j]['raw_energy']=np.array([stats.sumsq,stats.samples],np.float64);features[j]['native_sr']=row[4]
                    for mel in ([48] if sr==16000 else [48,64]):
                        matrix=front[(sr,mel)].mel;linear=power@matrix.T;peak_log=(linear/peak[:,None,None].square()).clamp_min(1e-10).log();linear_cpu=linear.cpu().numpy();peak_cpu=peak_log.cpu().numpy()
                        for j,n_samples in enumerate(lengths):
                            n=(n_samples-f.n_fft)//f.hop+1;features[j][f'power{sr}_mel{mel}']=linear_cpu[j,:n].astype(np.float32);features[j][f'peak{sr}_mel{mel}']=peak_cpu[j,:n].astype(np.float32)
                del x,frames,power;torch.cuda.empty_cache()
            for row,values in zip(batch,features):
                dest=folder/f'{row[0]:05d}.npz';temp=dest.with_suffix('.tmp')
                with temp.open('wb') as out:np.savez_compressed(out,**values)
                temp.replace(dest)
            if start%256==0:print('NEW_A1_TRAIN_CACHE',start+len(batch),'/',len(missing),'seconds',time.monotonic()-started,flush=True)
    assert len(list(folder.glob('*.npz')))==11020
    (ROOT/'FRONTEND_CACHE_COMPLETE_v3.json').write_text(json.dumps({'time':pd.Timestamp.now().isoformat(),'n':11020,'seconds_this_resume':time.monotonic()-started,'branches':['16000mel48','48000mel48','48000mel64'],'feature_identity':'linear mel-power plus peak controls, raw4level+4band finite running sides','side_columns':LEVEL_COLUMNS+BAND_COLUMNS,'VAL_audio':0,'TEST_audio':0},indent=2))
    note('新A1真实前端cache11020完成，TRAIN only，16k48mel/48k48mel/48k64mel+运行8侧信息，不使用旧THCHS100k。')
def load_features(d,config):
    from feature_store_v3 import FeatureStore
    folder=ROOT/'cache_v3';assert pd.read_csv(folder/'index.csv').file.tolist()==d.file.tolist()
    key=f'{"peak" if config["normalization"]=="peak" else "power"}{config["sr"]}_mel{config["n_mels"]}';packed=folder/'packed'/key;packed.mkdir(parents=True,exist_ok=True)
    if not (packed/'COMPLETE.json').exists():
        seq=[];side=[];energy=[]
        for i in range(len(d)):
            with np.load(folder/f'{i:05d}.npz') as a:seq.append(a[key]);side.append(a['side']);energy.append(a['raw_energy'])
        nmax=max(map(len,seq));x=np.zeros((len(d),nmax,config['n_mels']),np.float32);mask=np.zeros((len(d),nmax),bool)
        for i,z in enumerate(seq):x[i,:len(z)]=z;mask[i,:len(z)]=True
        for name,array in [('x',x),('mask',mask),('side',np.asarray(side,np.float32)),('raw_energy',np.asarray(energy,np.float64))]:np.save(packed/f'{name}.npy',array)
        (packed/'COMPLETE.json').write_text(json.dumps({'files':d.file.tolist(),'shape':list(x.shape),'key':key}))
    else:assert json.loads((packed/'COMPLETE.json').read_text())['files']==d.file.tolist()
    ids=(list(range(4)) if config['normalization']=='global_gain' else [])+(list(range(4,8)) if config['bandwidth_side'] else [])
    side=torch.tensor(np.load(packed/'side.npy')[:,ids]) if ids else None
    return FeatureStore(str(packed),config['normalization']),None,side
if __name__=='__main__':main()
