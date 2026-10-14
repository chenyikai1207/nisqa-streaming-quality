"""GPU training with per-epoch resumable state. v1 reads never mutate its artifacts."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,threading
import numpy as np
import pandas as pd
import torch
from audit_frontend_v3 import note,ROOT,DIMS
INIT_LOCK=threading.Lock()
NORMALIZER_LOCK=threading.Lock()

def cache_v1_gpu(d):
    assert pd.read_csv(ROOT/'train_cache_index.csv').file.tolist()==d.file.tolist()
    seq=[np.load(ROOT/'cache'/f'{i:05d}.npz')['logmel'] for i in range(len(d))]
    nmax=max(map(len,seq));x=np.zeros((len(d),nmax,48),np.float32);mask=np.zeros((len(d),nmax),bool)
    for i,z in enumerate(seq):x[i,:len(z)]=z;mask[i,:len(z)]=True
    del seq
    return torch.as_tensor(x,device='cuda'),torch.as_tensor(mask,device='cuda')

def normalizer(x,mask,ids):
    # Same masked population mean/variance, GPU chunks bound temporary residency.
    total=torch.zeros(x.shape[-1],device='cuda');count=mask[ids].sum().float()
    for batch in ids.split(64):total+=(x[batch]*mask[batch,:,None]).sum((0,1))
    mean=total/count;total.zero_()
    for batch in ids.split(64):total+=((x[batch]-mean).square()*mask[batch,:,None]).sum((0,1))
    return mean,(total/count).sqrt().clamp_min(1e-5)

def normalizer_exact(x,mask,ids):
    values=x[ids].clone(); valid=mask[ids,:,None]; count=valid.sum().float()
    mean=(values*valid).sum((0,1))/count
    values.sub_(mean).square_().mul_(valid)
    std=(values.sum((0,1))/count).sqrt().clamp_min(1e-5)
    return mean,std

def fit(x,mask,target,d,train_ids,eval_ids,model_factory,seed,folder,epochs=60,early_stop=True,side=None,normalizer_key=None,external=None,uses_teacher=True,exact_normalizer=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():return np.load(folder/'pred.npy'),json.loads((folder/'receipt.json').read_text())
    if hasattr(x,'resolve'):x,mask=x.resolve(train_ids)
    if side is not None:side=side.to('cuda')
    note(f'开始拟合{folder.relative_to(ROOT)}：seed{seed}、TRAIN={len(train_ids)}、eval={len(eval_ids)}、max_epochs={epochs}、early_stop={early_stop}。')
    torch.cuda.empty_cache()
    frontend_gain=float(getattr(x,'frontend_gain',1.));ti=torch.as_tensor(train_ids,device='cuda');vi=torch.as_tensor(eval_ids,device='cuda')
    last=folder/'last.pt';ck=None
    if last.exists():
        ck=torch.load(last,map_location='cuda',weights_only=False);mean=ck['mean'];std=ck['std'];side_mean=ck['side_mean'];side_std=ck['side_std']
    else:
        path=ROOT/'normalizers_v3'/f'{normalizer_key}.pt' if normalizer_key else None
        with NORMALIZER_LOCK:
            if path is not None and path.exists():
                norm=torch.load(path,map_location='cuda',weights_only=False);assert np.array_equal(norm['train_ids'],np.asarray(train_ids));mean=norm['mean'];std=norm['std'];side_mean=norm['side_mean'];side_std=norm['side_std']
            else:
                mean,std=normalizer_exact(x,mask,ti) if exact_normalizer else normalizer(x,mask,ti);side_mean=side_std=None
                if side is not None:
                    side_mean=torch.nan_to_num(torch.nanmean(side[ti],dim=0));ss=torch.where(torch.isfinite(side[ti]),side[ti],side_mean);side_std=ss.std(0,correction=0);side_std=torch.where(side_std>1e-5,side_std,torch.ones_like(side_std))
                if path is not None:
                    path.parent.mkdir(parents=True,exist_ok=True);torch.save({'mean':mean.cpu(),'std':std.cpu(),'side_mean':side_mean.cpu() if side_mean is not None else None,'side_std':side_std.cpu() if side_std is not None else None,'train_ids':np.asarray(train_ids)},path)
    torch.cuda.empty_cache()
    with INIT_LOCK:
        torch.manual_seed(seed);model=model_factory().cuda()
    order_rng=torch.Generator(device='cuda').manual_seed(seed)
    external_rng=np.random.default_rng(seed+20261011)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    y=torch.as_tensor(target,device='cuda',dtype=torch.float32);weight=torch.tensor([1.,.5,.5,.5,.5],device='cuda')
    start_epoch=0;best=float('inf');patience=0;history=[];elapsed=0.
    if ck is not None:
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start_epoch=ck['epoch'];best=ck['best'];patience=ck['patience'];history=ck['history'];elapsed=ck['elapsed'];order_rng.set_state(ck.get('order_rng',ck['rng_cuda'][0]).cpu())
        if 'external_rng' in ck:external_rng.bit_generator.state=ck['external_rng']
        note(f'恢复{folder.relative_to(ROOT)}：完整epoch{start_epoch}，未重复已完成epoch。')
    def forward(ids):
        nn=int(mask[ids].sum(1).max().item());z=((x[ids,:nn]-mean)/std)*mask[ids,:nn,None]
        if side is None:return model(z,mask[ids,:nn])
        ss=torch.where(torch.isfinite(side[ids]),side[ids],side_mean)
        return model(z,mask[ids,:nn],(ss-side_mean)/side_std)
    st=time.perf_counter()
    for epoch in range(start_epoch,epochs):
        epoch_start=time.perf_counter()
        cpu_start=time.process_time();gpu_start=torch.cuda.Event(enable_timing=True);gpu_train_end=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        model.train();order=ti[torch.randperm(len(ti),device='cuda',generator=order_rng)];loss_sum=0
        for ids in order.split(128):
            opt.zero_grad(set_to_none=True);pred=forward(ids);good=torch.isfinite(y[ids]);loss=((pred-torch.nan_to_num(y[ids])).square()*good*weight).sum()/(good*weight).sum()
            loss_value=loss.item();loss.backward()
            if external is not None and external.n and external.alpha<1:
                ei=external_rng.integers(0,external.n,size=len(ids));ex,em,ey,es=external.batch(ei)
                ex=((ex-mean)/std)*em[:,:,None]
                ep=model(ex,em) if side is None else model(ex,em,(es-side_mean)/side_std)
                eg=torch.isfinite(ey);eloss=((ep-torch.nan_to_num(ey)).square()*eg*weight).sum()/(eg*weight).sum()
                scaled=(1-external.alpha)*eloss;scaled.backward();loss_value+=scaled.item()
            opt.step();loss_sum+=loss_value*len(ids)
        scheduler.step()
        gpu_train_end.record();train_wall=time.perf_counter()-epoch_start
        value=None
        if early_stop:
            model.eval();pp=[]
            with torch.inference_mode():
                for ids in vi.split(128):pp.append(forward(ids).cpu().numpy())
            pred=np.concatenate(pp);sub=d.iloc[np.asarray(eval_ids)];doms=sub.domain.unique();value=float(np.mean([np.abs(sub.loc[sub.domain.eq(dom),'mos'].to_numpy()-pred[sub.domain.eq(dom),0]).mean() for dom in doms]))
            if value<best-1e-7:
                best=value;patience=0
                torch.save({'state_dict':model.state_dict(),'frontend_gain':frontend_gain,'mean':mean.cpu(),'std':std.cpu(),'side_mean':side_mean.cpu() if side_mean is not None else None,'side_std':side_std.cpu() if side_std is not None else None,'epoch':epoch+1},folder/'best.pt');np.save(folder/'pred.npy',pred)
            else:patience+=1
        gpu_end.record();gpu_end.synchronize()
        history.append(dict(epoch=epoch+1,training_mse=loss_sum/len(ti),cv_mos_mae=value,epoch_seconds=time.perf_counter()-epoch_start,elapsed_seconds=elapsed+time.perf_counter()-st,training_wall_seconds=train_wall,validation_wall_seconds=time.perf_counter()-epoch_start-train_wall,cuda_training_seconds=gpu_start.elapsed_time(gpu_train_end)/1000,cuda_epoch_seconds=gpu_start.elapsed_time(gpu_end)/1000,process_cpu_seconds=time.process_time()-cpu_start,explicit_resource_wait_seconds=0.,allocated_MB=torch.cuda.memory_allocated()/1e6,reserved_MB=torch.cuda.memory_reserved()/1e6,worker_pid=os.getpid()))
        ck={'state_dict':model.state_dict(),'frontend_gain':frontend_gain,'mean':mean.cpu(),'std':std.cpu(),'side_mean':side_mean.cpu() if side_mean is not None else None,'side_std':side_std.cpu() if side_std is not None else None,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'patience':patience,'history':history,'elapsed':elapsed+time.perf_counter()-st,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'order_rng':order_rng.get_state()}
        ck['external_rng']=external_rng.bit_generator.state
        torch.save(ck,folder/'last.tmp');(folder/'last.tmp').replace(last)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print('EPOCH',str(folder.relative_to(ROOT)),seed,epoch+1,'/',epochs,'mse',loss_sum/len(ti),'cv',value,'gpu_MB',torch.cuda.memory_allocated()/1e6,flush=True)
        if early_stop and patience>=8:break
    if not early_stop:
        model.eval();pp=[]
        with torch.inference_mode():
            for ids in vi.split(128):pp.append(forward(ids).cpu().numpy())
        pred=np.concatenate(pp);np.save(folder/'pred.npy',pred)
        torch.save({'state_dict':model.state_dict(),'frontend_gain':frontend_gain,'mean':mean.cpu(),'std':std.cpu(),'side_mean':side_mean.cpu() if side_mean is not None else None,'side_std':side_std.cpu() if side_std is not None else None,'epoch':len(history)},folder/'best.pt')
    r={'seed':seed,'epochs':len(history),'best_epoch':int(pd.DataFrame(history).cv_mos_mae.idxmin()+1) if early_stop else len(history),'early_stop':early_stop,'parameters':sum(p.numel() for p in model.parameters()),'train_n':len(ti),'eval_n':len(vi),'TRAIN_groups':d.iloc[np.asarray(train_ids)].ref_group.nunique(),'seconds':elapsed+time.perf_counter()-st,'teacher_not_crossfit':bool(uses_teacher),'device':'cuda','time':pd.Timestamp.now().isoformat(),'normalizer_key':normalizer_key,'private_order_rng':True}
    free,total=torch.cuda.mem_get_info()
    r.update(external_n=external.n if external is not None else 0,external_alpha=external.alpha if external is not None else None,uses_teacher=bool(uses_teacher),teacher_exposure_if_used='official TRAIN/VAL development; not retrained inside student folds' if uses_teacher else None)
    r.update(frontend_gain=frontend_gain,worker_pid=os.getpid(),peak_allocated_MB=torch.cuda.max_memory_allocated()/1e6,peak_reserved_MB=torch.cuda.max_memory_reserved()/1e6,global_free_MB=free/1e6,global_total_MB=total/1e6,feature_residency='cpu_mmap_private_cuda_batches' if hasattr(x,'array') else 'private_cuda_full')
    (folder/'receipt.json').write_text(json.dumps(r,indent=2));note(f'完成{folder.relative_to(ROOT)}，epochs={r["epochs"]}、TRAIN={len(ti)}、eval={len(vi)}、参数={r["parameters"]}、PID={os.getpid()}、显存峰值allocated={r["peak_allocated_MB"]:.1f}MB/reserved={r["peak_reserved_MB"]:.1f}MB。')
    return np.load(folder/'pred.npy'),r

def fit_on_stream(*args,**kwargs):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):result=fit(*args,**kwargs)
    stream.synchronize();return result
