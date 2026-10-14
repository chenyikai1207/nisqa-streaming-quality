"""Manuscript cost tables and number sources from completed cost_v3 outputs only."""
from pathlib import Path
import os
import csv
import json

HERE = Path(__file__).resolve().parent
STAGE=Path(__file__).resolve().parent/'scientific_inputs/R4'
C = STAGE / 'exp_D/cost_v3'
R = C / 'results'
LABELS = {
    ('v1', 'native'): '首版 v1',
    ('v2_96', 'native'): '双层GRU96 原生',
    ('v2_96', 'deploy'): '双层GRU96 部署F32',
    ('v2_64', 'deploy'): '双层GRU64 部署F32',
    ('NISQA_MOS', 'official'): '官方NISQA MOS',
    ('PLCMOS', 'official'): 'PLCMOS',
}
ORDER = list(LABELS)

def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def truth(value):
    return value is True or str(value).lower() == 'true'

def ready():
    path=R/'cost_final.csv'
    if not path.exists():return False
    rows=csv_rows(path)
    return len(rows)==16 and all(truth(r['RSS_complete']) and truth(r['MAC_complete']) for r in rows)

def dataset():
    if not ready():
        raise RuntimeError('Cost measurement window still active; manuscript publication deferred')
    costs = csv_rows(R / 'cost_final.csv')
    stream = csv_rows(R / 'streaming_kernel.csv')
    mac = csv_rows(R / 'mac_table.csv')
    if len(costs) != 16 or any(not truth(row['RSS_complete']) for row in costs):
        raise RuntimeError('Final 16 cost/RSS rows not yet complete')
    if len(stream) != 6 or any(int(row['n_blocks']) != 1000 for row in stream):
        raise RuntimeError('Final six streaming timing/resident pairs not complete')
    rep = R / 'representativeness/ACTUAL_REPRESENTATIVENESS_COMPLETE.json'
    actual_rep = json.loads(rep.read_text(encoding='utf-8'))
    if len(actual_rep['results']) != 2:
        raise RuntimeError('Both actual fullVAL representativeness passes required')
    return costs, stream, mac

def fmt(value, digits=4):
    return '—' if value in ('', None) else f'{float(value):.{digits}f}'

def cost_tables():
    costs, stream, mac = dataset()
    by = {(row['method'], row['path'], row['setting']): row for row in costs}
    totals = {row['method']: row for row in mac if row['component'] == 'TOTAL'}
    inventory = json.loads((R / 'model_inventory.json').read_text(encoding='utf-8'))
    tables = []
    def add(suffix, title, headers, rows, note):
        tables.append(dict(number=8, cn=f'表8({suffix}) {title}', en=f'Table 8({suffix})',
                           headers=headers, rows=rows, note=note))
    rows = []
    for method, path in ORDER:
        key = method + '_' + path
        item = inventory[key]
        rows.append([LABELS[(method, path)], str(item['parameters']), str(item['weights_bytes']),
                     fmt(totals[key]['mmac_per_s'], 2 if method == 'PLCMOS' else 4)])
    for key, label in [('NISQA5D_official', '官方NISQA五维'),
                       ('DNSMOS_official', 'DNSMOS'), ('SQUIM_official', 'SQUIM')]:
        item = inventory[key]
        rows.append([label, str(item['parameters']), str(item['weights_bytes']), fmt(totals[key]['mmac_per_s'])])
    rows.append(['P.563', '—', '—', '—'])
    add('a', '参数、权重文件与完整前端运算估计', ['方法/路径', '参数', '权重文件/B', 'MMACeq/音频秒'], rows,
        '运算估计按VAL平均时长8.649216358 s与实际帧取整统一；包含前端、FFT估计、FIR上界和标量代理，非硬件指令精确计数。PLCMOS为一次外部STFT及15次完整网络调用。权重文件包括部署路径需要的冻结数组及FIR，不等同参数载荷或RSS。P.563不可得，选测方法没有新增正式RTF。')
    rows = []
    failures = []
    for method, path in ORDER:
        vals = [LABELS[(method, path)]]
        for setting in ['S3', 'S1']:
            row = by[(method, path, setting)]
            mark = '' if truth(row['accepted']) else '†'
            vals.extend([fmt(row['rtf_median'], 6) + mark,
                         fmt(float(row['iqr_ratio']) * 100, 3), fmt(float(row['calib_drift']) * 100, 3)])
            if mark:
                failures.append(f"{LABELS[(method,path)]}/{setting}：离散度{100*float(row['iqr_ratio']):.3f}%、最大校准漂移{100*float(row['calib_drift']):.3f}%")
        rows.append(vals)
    add('b', '同一600条VAL的热缓存端到端RTF',
        ['方法/路径', 'S3 RTF', 'S3 IQR/%', 'S3漂移/%', 'S1 RTF', 'S1 IQR/%', 'S1漂移/%'], rows,
        '每项20文件预热、5个独立进程；S3固定E16单线程，S1不绑核且库默认线程，Torch实际16个算子内线程、可调度范围24逻辑核。基础应用背景下受控测量，核心温度不可用。16设置中8个通过全部标准，8个未同时通过预定稳定性标准并加†，原始测量保留；每项最多两次额外重测。IQR/%为四分位距/中位数；漂移为块及整轮前后校准最大相对差。' + '；'.join(failures))
    rows = []
    for key in [('v1', 'native'), ('v2_96', 'native'), ('v2_96', 'deploy'), ('NISQA_MOS', 'official')]:
        row = by[(*key, 'S3')]
        rows.append([LABELS[key]] + [fmt(row['t_' + name], 3) for name in ['L', 'R', 'G', 'F', 'I', 'O']])
    add('c', '单能效核完整600文件的阶段耗时', ['方法/路径', 'L/s', 'R/s', 'G/s', 'F/s', 'I/s', 'O/s'], rows,
        'L读取解码，R重采样，G电平，F前端，I推理，O其余开销。各阶段分别取五轮中位数，和不必等于端到端中位数。NISQA内部前端及推理不可分部分合并于I；0表示该阶段未单列，不表示物理操作没有成本。采用表8(b)同一最终完整组，未达标标记同表。')
    rows = []
    for key in ORDER:
        vals = [LABELS[key]]
        for setting in ['S3', 'S1']:
            row = by[(*key, setting)]
            vals.extend([fmt(row['baseline_rss_mb'], 4), fmt(row['peak_rss_mb'], 4)])
        rows.append(vals)
    add('d', '独立整句进程内存测量', ['方法/路径', 'S3基线/MB', 'S3峰值/MB', 'S1基线/MB', 'S1峰值/MB'], rows,
        '每设置独立处理同600文件，以10 ms间隔采样；基线在库/模型加载后、第一文件前。MB=10^6 B，含解释器、库、模型及音频缓冲；与精简流式RSS及状态载荷分别报告。')
    main = [row for row in mac if row['method'] == 'v2_96_deploy' and row['component'] != 'TOTAL']
    add('e', '主部署路径的完整运算组成', ['组件', 'MMACeq/音频秒'],
        [[row['component'], fmt(row['mmac_per_s'], 6)] for row in main],
        '与表8(a)同一新VAL平均时长口径。解析矩阵乘加、FFT估计、FIR上界及标量代理分项保留；未执行的分支不计入部署路径。')
    rows = []
    for row in stream:
        rows.append(['GRU96' if row['method'] == 'v2_pure' else 'GRU64', row['chunk_ms'],
                     fmt(row['block_ms_median'], 4), fmt(row['block_ms_p99'], 4), row['state_bytes'],
                     fmt(row['resident_rss_mb'], 4), fmt(row['algorithm_delay_ms'], 4)])
    add('f', '精简F32流式内核的块耗时、载荷和常驻内存',
        ['模型', '块/ms', '中位/ms', 'P99/ms', '观察状态/B', 'RSS/MB', '时延/ms'], rows,
        'E16单线程，固定48 kHz TRAIN单流循环输入，每种块长1000块；timing/resident独立进程，实际NumPy后端，无Torch/ORT。观察状态为本输入的保留数组载荷及64位逻辑寄存器；各实际输入采样率的解析上界另见表8(j)。权重、块内暂存及Python对象由进程RSS反映。RSS为独立resident进程中位数。P99是第99百分位，不是最大块耗时。算法时延=块长+0.625 ms FIR，未来前瞻0；20 ms谱窗累积和首输出所需输入时长另列。')
    rows = []
    for key in [('v1', 'native'), ('v2_96', 'native'), ('v2_96', 'deploy'), ('NISQA_MOS', 'official')]:
        row = by[(*key, 'S2')]
        rows.append([LABELS[key], fmt(row['rtf_median'], 6), fmt(float(row['iqr_ratio'])*100, 3),
                     fmt(float(row['calib_drift'])*100, 3), '通过' if truth(row['accepted']) else '未通过'])
    add('h', '单性能核补充测量', ['方法/路径', 'RTF', 'IQR/%', '漂移/%', '验收'], rows,
        'P2固定单线程，与S3同一600列表；四设置均按事前规则验收。补充口径不替代S3摘要主口径。')
    representative = csv_rows(R/'representativeness/representativeness.csv')
    add('i', '完整VAL单次代表性核对', ['方法/路径', '文件数', '完整VAL RTF', '600条RTF', '相对差/%', '<5%'],
        [[LABELS[(row['method'], row['path'])], row['n_files'], fmt(row['rtf_fullVAL_single'], 6),
          fmt(row['rtf_bench600_fiveround_median'], 6), fmt(float(row['relative_difference'])*100 if row['relative_difference'] != '' else '', 3),
          '是' if truth(row['within5percent']) else '否'] for row in representative],
        '固定S3，计划各2700文件，实际完成范围按行列示，未完成项记—；相对差=|完整VAL单次RTF/600条五轮中位数−1|。不替换600主口径，未达5%也不追加测量。')
    bounds_path = R/'stream_state_bounds_by_samplerate.csv'
    if bounds_path.exists():
        bounds = csv_rows(bounds_path)
        add('j', '实际输入采样率下的流式状态载荷上界', ['模型','输入采样率/Hz','状态上界/B'],
            [['GRU96' if r['method']=='v2_pure' else 'GRU64', r['input_sr'], r['state_bytes_upper'] or '不支持'] for r in bounds],
            '按精简后端实际dtype的GRU隐状态、增量池化、梅尔与侧路残留、两个因果重采样器的历史和逻辑计数解析计算；覆盖初始化、残留长度及重采样相位。完整11020条TRAIN的音频头均为48 kHz；16 kHz由现有滤波核支持但未在此TRAIN库存出现，8/32 kHz未由现有核支持。保留状态包含数组载荷及27个64位逻辑寄存器；权重、块内暂存、系数表与Python对象分别由权重大小和进程RSS报告。')
    return tables

def fill_cost_numbers(put):
    costs, stream, mac = dataset()
    by = {(r['method'], r['path'], r['setting']): r for r in costs}
    totals = {r['method']: r for r in mac if r['component'] == 'TOTAL'}
    for key, row in by.items():
        selector = dict(zip(['method', 'path', 'setting'], key))
        prefix = 'D.cost.' + '.'.join(key)
        for col in ['rtf_median', 'iqr_ratio', 'calib_drift', 'baseline_rss_mb', 'peak_rss_mb', 'cold_s']:
            put(prefix + '.' + col, float(row[col]), R/'cost_final.csv', selector, col,
                6 if col in ['rtf_median','iqr_ratio','calib_drift'] else 4)
        put(prefix + '.t_I', float(row['t_I']), R/'cost_final.csv', selector, 't_I', 4)
    main_stream = next(r for r in stream if r['method'] == 'v2_pure' and r['chunk_ms'] == '20')
    selector = dict(method='v2_pure', chunk_ms=20)
    for col in ['resident_rss_mb', 'state_bytes', 'block_ms_median', 'block_ms_p99', 'algorithm_delay_ms', 'first_prediction_input_ms']:
        put('D.cost.stream20.' + col, float(main_stream[col]), R/'streaming_kernel.csv', selector, col,
            0 if col == 'state_bytes' else 4)
    for row in stream:
        for col in ['resident_rss_mb','block_ms_median','block_ms_p99','block_rtf_p99']:
            put('D.cost.stream.'+row['method']+'.'+row['chunk_ms']+'.'+col, float(row[col]),
                R/'streaming_kernel.csv',dict(method=row['method'],chunk_ms=row['chunk_ms']),col,4)
    bound_path=R/'stream_state_bounds_by_samplerate.csv'
    if bound_path.exists():
        bounds=csv_rows(bound_path)
        for method in ['v2_pure','v2_64']:
            upper=max(int(r['state_bytes_upper']) for r in bounds if r['method']==method and str(r['state_bytes_upper']).isdigit())
            put('D.cost.state_bound.'+method+'.state_bytes_upper',upper,bound_path,dict(method=method),'max state_bytes_upper',0)
            put('D.cost.state_bound.'+method+'.state_kB_upper',upper/1000,bound_path,dict(method=method),'max state_bytes_upper / 1000',3)
    for method in ['v1_native', 'v2_96_native', 'v2_96_deploy', 'v2_64_deploy', 'NISQA_MOS_official', 'PLCMOS_official']:
        put('D.cost.mac.' + method, float(totals[method]['mmac_per_s']), R/'mac_table.csv',
            dict(method=method, component='TOTAL'), 'mmac_per_s', 2 if method == 'PLCMOS_official' else 4)
    native_mac = float(totals['v2_96_native']['mmac_per_s'])
    put('D.main.total_mac_estimate', f'{native_mac:.4f}百万MAC等价量/音频秒', R/'mac_table.csv',
        dict(method='v2_96_native', component='TOTAL'), 'mmac_per_s')
    put('D.main.mac_million', native_mac, R/'mac_table.csv', dict(method='v2_96_native', component='TOTAL'), 'mmac_per_s')
    put('D.main.s3_rtf', float(by[('v2_96','deploy','S3')]['rtf_median']), R/'cost_final.csv',
        dict(method='v2_96', path='deploy', setting='S3'), 'rtf_median', 6)
    put('D.cost.s2.nisqa_ratio', float(by[('NISQA_MOS','official','S2')]['rtf_median']) /
        float(by[('v2_96','deploy','S2')]['rtf_median']), R/'cost_final.csv',
        dict(reference='NISQA_MOS/official/S2', target='v2_96/deploy/S2'), 'rtf_median ratio', 2)

if __name__ == '__main__':
    tables = cost_tables()
    (HERE/'cost_v3_tables.json').write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='ACTUAL_COST_V3_TABLES', tables=len(tables), rows=sum(len(t['rows']) for t in tables))))
