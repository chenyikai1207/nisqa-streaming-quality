"""Reader-facing eight-table projection of completed owner outputs."""
from copy import deepcopy
import json

LABELS={'GRU真实标签':'首版（无蒸馏）','GRU蒸馏v1':'首版','GRU v1':'首版','最终模型 v2':'本文模型','GRU v2':'本文模型','首版 v1':'首版','双层GRU96 原生':'本文模型 原生','双层GRU96 部署F32':'本文模型 整句F32','双层GRU64 部署F32':'GRU64 整句F32'}
def lab(x):return LABELS.get(str(x),str(x))

def compose(tables,here):
    old=deepcopy(tables)
    def get(prefix):
        matches=[t for t in old if t['cn'].startswith(prefix)]
        if len(matches)!=1:raise ValueError('Eight-table source selector '+prefix)
        return deepcopy(matches[0])
    def item(number,cn,en,headers,rows,note):
        return dict(number=number,cn=cn,en=en,headers=headers,rows=rows,note=note)
    main=[]
    t=get('表1 ');t.update(number=1,cn='表1 数据与分组',en='Table 1. Data and grouping');main.append(t)
    t=get('表2(b)');t.update(number=2,cn='表2 结构与容量的MOS训练结果',en='Table 2. MOS training results across architectures and capacities',note='三个训练种子均值±SD；相同16 kHz、48梅尔前端。按4.1的综合分和成本规则选出双层GRU96，*表示该结构。SD为训练初始化变化。');main.append(t)
    ma=get('表2(c)');pr=get('表2(d)');rows=[]
    for a,b in zip(ma['rows'],pr['rows']):
        if a[:2]!=b[:2]:raise ValueError('Frontend panel row identity differs')
        rows.append([*a[:2],a[2]+'→'+a[3],a[4],b[2]+'→'+b[3],b[4]])
    main.append(item(3,'表3 双层GRU96前端A/B的MAE与相关性','Table 3. MAE and correlation in the two-layer GRU96 frontend A/B comparison',['训练设计','维度','MAE A→B','A−B下降量 [95%CI]','Pearson A→B','B−A增量 [95%CI]'],rows,'A：整段峰值归一化、无原始电平四统计；B：训练侧固定增益、原始电平四统计。先平均三个种子的逐文件预测，再计算指标；参考组分层同步bootstrap 2000次。MAE下降量和Pearson增量均以正值为改善；SIM→LIVE MOS/LOUD用于表述判断。'))
    quality=[get(x) for x in ['表3 ','表4(a)','表4(b)','表4(c)','表4(d)']]
    labels=[]
    for t in quality:
        for r in t['rows']:
            if r[0] not in labels and r[0] not in ['双层GRU64','卷积k5+GRU64']:labels.append(r[0])
    rows=[]
    for label in labels:
        rr=[lab(label)]
        for t in quality:
            found=[r for r in t['rows'] if r[0]==label]
            if not found:rr.extend(['—','—']);continue
            r=found[0];rmse=r[5]
            if label=='UTMOSv2':rmse=r[4]+'ᵇ'
            rr.extend([r[2],rmse])
        rows.append(rr)
    main.append(item(4,'表4 各方法在VAL及跨集评价中的MOS精度','Table 4. MOS accuracy on VAL and cross-dataset evaluations',['方法','VAL\nr','VAL\nRMSE','FOR\nr','FOR\nRMSE','P501\nr','P501\nRMSE','LIVE\nr','LIVE\nRMSE','TCD\nr','TCD\nRMSE'],rows,'r为逐文件原始Pearson；RMSE采用既定TRAIN映射，官方NISQA保持原值。ᵇ UTMOSv2未拟合TRAIN映射，列原值RMSE。—表示该评价范围无预测；P.563使用实际有效样本。DNSMOS OVRL和SQUIM PESQ为不同目标的对比输出。'))
    t=get('表4(f)');t.update(number=5,cn='表5 本文模型对三个比较者的MOS文件层配对结果',en='Table 5. File-level paired MOS comparisons against three comparators');t['rows']=[[r[0],lab(r[1]),*r[2:]] for r in t['rows']];main.append(t)
    t=get('表5(b)');t.update(number=6,cn='表6 学习检测特征对MOS和DIS的增量',en='Table 6. MOS and DIS increments from learned detection features',note='44维基础组合加入8维检测特征，总52维。同容量HGB、三个训练种子逐文件绝对误差平均；参考组配对bootstrap 2000次。下降量为正表示MAE降低。');main.append(t)
    dsp=get('表7(a)');cnn=get('表7(b)');rows=[]
    for r in dsp['rows']:rows.append(['无噪声探针',r[0],r[1],r[2],'—',r[3],r[4],r[5]])
    for r in cnn['rows']:rows.append([r[0]+'留出',r[1],'—',r[4],r[3],r[5],r[6],str(r[7])+'/'+str(r[2])])
    main.append(item(7,'表7 补偿线索可观测性与两来源留出表现','Table 7. Concealment-cue observability and two-source held-out performance',['来源/探针','补偿','旧80ms ρ [CI]','DSP帧F1','CNN帧F1','Ppl MAE','BurstR ρ','联合n/全部n'],rows,'无噪声探针行的Ppl和ρ来自20 ms对齐DSP；来源留出行来自固定CNN，DSP列为零检测对照。ρ在联合可定义子集内计算，分母同时给出；20,000条留出仿真按来源及补偿分层。旧80 ms声学链与真实包链的响应区分报告。'))
    mac=get('表8(a)');s3={r[0]:r for r in get('表8(b)')['rows']};s2={r[0]:r for r in get('表8(h)')['rows']}
    stream={r[0]:r for r in get('表8(f)')['rows'] if str(r[1])=='10'}
    bounds={}
    for r in get('表8(j)')['rows']:
        if str(r[1])=='48000':bounds[r[0]]=r[2]
    rows=[]
    for r in mac['rows']:
        name=r[0];st=stream.get('GRU96' if name=='双层GRU96 部署F32' else 'GRU64' if name=='双层GRU64 部署F32' else '')
        state=bounds.get('GRU96' if name=='双层GRU96 部署F32' else 'GRU64' if name=='双层GRU64 部署F32' else '', '—')
        rows.append([lab(name),r[1],r[3],s2[name][1] if name in s2 else '—',s3[name][1] if name in s3 else '—',st[3] if st else '—',st[4] if st else '—',state])
    main.append(item(8,'表8 参数、完整运算与整句及流式成本','Table 8. Parameters, full operation estimates, and utterance/streaming costs',['模型/路径','参数','MMACeq/秒','S2 RTF','S3 RTF','10ms P99/ms','流式RSS/MB','状态上界/B'],rows,'S2为单性能核，S3为单能效核；固定600条VAL、五轮热缓存端到端中位数。†表示环境标准未通过，测量值保留。10 ms块的P99、RSS来自独立精简流式内核；状态上界覆盖支持的16/48 kHz输入。其他采样率需额外重采样状态。模型路径、阶段耗时、IQR/漂移、权重文件、各块长及逐率状态详见补充表。DNSMOS原TRAIN200时长加权与VAL代表长度估计口径不同，作者9.01 s短音重复和1 s离散窗口、固定STFT卷积均计入。'))
    moved_prefixes=['表2(a)','表2(e)','表5(a)','表5(c)','表6 ','表7(c)','表7(d)','表7(e)','表7(f)','表7(g)']+[f'表8({x})' for x in 'abcdefghij']
    moved=[]
    for n,prefix in enumerate(moved_prefixes,start=10):
        t=deepcopy(get(prefix));t['id']='S'+str(n)
        cn=t['cn'].split(' ',1)[1];en=t['en'].split(' ',2)[-1]
        t['title']=f'表S{n} '+cn+f'\nTable S{n}. '+en
        if '44维由' in t.get('note',''):t['note']=t['note'].split('44维由')[0].rstrip()
        t['rows']=[[lab(r[0]),*r[1:]] for r in t['rows']]
        t.pop('number',None);t.pop('cn',None);t.pop('en',None)
        moved.append(t)
    (here/'reader_moved_tables.json').write_text(json.dumps(moved,ensure_ascii=False,indent=2),encoding='utf-8')
    return main

