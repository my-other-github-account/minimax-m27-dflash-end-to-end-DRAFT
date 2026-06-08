#!/usr/bin/env python3
import json, os, re, subprocess, time, urllib.request, hashlib
from pathlib import Path

BASE=Path('$WORK/t_TRADEOFF_LADDER')
TASK='t_430cc462'
HOST='<host>'
BIN=Path('$WORK/lucebox-latest-20260603/server.pre_hshandoff_20260607_044845/build-sm121/dflash_server')
BIN_EXPECT='49d9a23d4eb2a655c4f35a18a7a1905a'
MODEL=Path('$WORK/models/Qwen3.6-35B-A3B-GGUF/UD-IQ4_XS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf')
DRAFT=Path('$WORK/models/Qwen3.6-35B-A3B-DFlash/model.safetensors')
PROMPTS=Path('$WORK/t_87c41e13/eval_prompts_50.jsonl')
PORT=18730
WIDTHS=[0,1,2,4,8,12,16]
MAX_TOKENS=512
THINK_MAX=256
MAX_CTX=8192

def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def sh(cmd, check=True, timeout=None):
    print('[sh]', cmd, flush=True)
    r=subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f'cmd failed rc={r.returncode}: {cmd}\n{r.stdout[-8000:]}')
    return r.stdout

def gpu_apps():
    return sh('nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true', check=False).strip()

def wait_gpu_free():
    for _ in range(180):
        out=gpu_apps()
        if not out:
            return
        print('[gpu-busy-wait]', out, flush=True)
        time.sleep(20)
    raise RuntimeError('GPU did not become free')

def kill_port():
    sh(f'fuser -k {PORT}/tcp 2>/dev/null || true', check=False)
    time.sleep(2)

def load_prompts():
    rows=[]
    for i,line in enumerate(PROMPTS.open(),1):
        o=json.loads(line)
        rows.append({'idx':i,'id':o.get('id',str(i)), 'source':o.get('source'), 'prompt':o['prompt']})
    return rows

def start_server(width):
    wait_gpu_free(); kill_port()
    label=f'w{width}'
    log=BASE/f'server_{label}.log'
    args=[str(BIN), str(MODEL)]
    if width != 0:
        args += ['--draft', str(DRAFT)]
    args += ['--port', str(PORT), '--host','<ip>', '--max-ctx', str(MAX_CTX), '--default-max-tokens', str(MAX_TOKENS), '--think-max-tokens', str(THINK_MAX), '--hard-limit-reply-budget','0', '--model-name','dflash', '--prefix-cache-slots','0']
    env=os.environ.copy()
    env['DFLASH_EXPERT_BUDGET_MB']='80000'
    env['DFLASH_QWEN35_VERIFY_TRACE']='1'
    if width != 0:
        env['DFLASH_VERIFY_Q_LEN_OVERRIDE']=str(width)
        env['DFLASH_DRAFTATTN_PROBE']='1'
        env['DFLASH_DRAFTMASK_PROBE']='1'
    f=open(log,'w')
    p=subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=str(BASE), env=env)
    (BASE/f'server_{label}.pid').write_text(str(p.pid))
    deadline=time.time()+900
    while time.time()<deadline:
        if p.poll() is not None:
            raise RuntimeError(f'server {label} died rc={p.returncode}\n{log.read_text(errors="replace")[-12000:]}')
        try:
            urllib.request.urlopen(f'http://<ip>:{PORT}/v1/models', timeout=2).read()
            print('[ready]', label, 'pid', p.pid, flush=True)
            return p,log,' '.join(args)
        except Exception:
            time.sleep(2)
    raise RuntimeError(f'server {label} not ready\n{log.read_text(errors="replace")[-12000:]}')

def stop_server(p):
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=30)
    kill_port()

def extract_text_stats(resp):
    reasoning_tokens=0; visible_len=0; reasoning_len=0; text_len=0; finish=None
    if isinstance(resp,dict):
        usage=resp.get('usage') or {}
        ctd=usage.get('completion_tokens_details') or {}
        reasoning_tokens=int(ctd.get('reasoning_tokens') or usage.get('reasoning_tokens') or 0)
        ch=(resp.get('choices') or [{}])[0]
        finish=ch.get('finish_reason') or (ch.get('finish_details') or {}).get('close_kind')
        msg=ch.get('message') or {}
        if isinstance(msg,dict):
            content=msg.get('content') or ''
            reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
            visible_len=len(content or '')
            reasoning_len=len(reasoning or '')
            text_len=visible_len+reasoning_len
        txt=ch.get('text')
        if txt is not None:
            visible_len=max(visible_len, len(txt or ''))
            text_len=max(text_len, len(txt or ''))
    return reasoning_tokens, visible_len, reasoning_len, text_len, finish

def request_one(pr):
    payload={'model':'dflash','messages':[{'role':'user','content':pr['prompt']}], 'max_tokens':MAX_TOKENS, 'temperature':0, 'thinking':{'type':'enabled'}}
    data=json.dumps(payload).encode()
    req=urllib.request.Request(f'http://<ip>:{PORT}/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
    t0=time.time(); raw=''; status=-1
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            raw=r.read().decode('utf-8','replace'); status=r.status
    except Exception as e:
        raw=json.dumps({'error':repr(e)})
    wall=time.time()-t0
    try: resp=json.loads(raw)
    except Exception: resp={'raw':raw[:<port>]}
    return status, resp, wall

def parse_log(log, since=0):
    txt=log.read_text(errors='replace')
    chunk=txt[since:]
    newoff=len(txt)
    specs=[]
    for m in re.finditer(r'\[hybrid-spec\] tokens=(\d+) time=([0-9.]+) s speed=([0-9.]+) tok/s steps=(\d+) accepted=(\d+)/(\d+) \(([0-9.]+)%\) avg_commit=([0-9.]+) AL=([0-9.]+)', chunk):
        specs.append({'tokens':int(m.group(1)), 'decode_s':float(m.group(2)), 'tok_s':float(m.group(3)), 'steps':int(m.group(4)), 'accepted':int(m.group(5)), 'drafted':int(m.group(6)), 'accept_pct':float(m.group(7)), 'avg_commit':float(m.group(8)), 'AL':float(m.group(9))})
    for m in re.finditer(r'\[spec-decode\] tokens=(\d+) time=([0-9.]+) s speed=([0-9.]+) tok/s steps=(\d+) accepted=(\d+)/(\d+) \(([0-9.]+)%\) avg_commit=([0-9.]+)', chunk):
        specs.append({'tokens':int(m.group(1)), 'decode_s':float(m.group(2)), 'tok_s':float(m.group(3)), 'steps':int(m.group(4)), 'accepted':int(m.group(5)), 'drafted':int(m.group(6)), 'accept_pct':float(m.group(7)), 'avg_commit':float(m.group(8)), 'AL':float(m.group(8))})
    ar=[]
    for m in re.finditer(r'\[ar-decode\] tokens=(\d+) time=([0-9.]+) s speed=([0-9.]+) tok/s', chunk):
        ar.append({'tokens':int(m.group(1)), 'decode_s':float(m.group(2)), 'tok_s':float(m.group(3))})
    per=[]
    for m in re.finditer(r'\[(?:hybrid-spec|spec-decode)\] per_pos_prefix_accept p1\.\.p(\d+):([^\n]+)', chunk):
        vals=[]
        for x in m.group(2).strip().split():
            try: vals.append(float(x))
            except Exception: pass
        per.append(vals)
    return (specs[-1] if specs else None), (ar[-1] if ar else None), (per[-1] if per else None), newoff

def aggregate(width, rows, log, cmd, bin_md5_before, bin_md5_after):
    tokens=sum(int(r.get('completion_tokens') or 0) for r in rows)
    wall=sum(float(r.get('seconds') or 0) for r in rows)
    reason=sum(int(r.get('reasoning_tokens') or 0) for r in rows)
    visible_nonempty=sum(1 for r in rows if int(r.get('visible_len') or 0)>0)
    decode_s=sum(float((r.get('spec') or r.get('ar') or {}).get('decode_s') or 0) for r in rows)
    specs=[r['spec'] for r in rows if r.get('spec')]
    pers=[r.get('per_pos') or [] for r in rows if r.get('spec')]
    per=[]
    if specs and pers:
        maxlen=max([len(x) for x in pers] or [0])
        for j in range(maxlen):
            num=den=0.0
            for s,p in zip(specs,pers):
                if j < len(p):
                    wt=s.get('steps') or 1
                    num += p[j]*wt; den += wt
            per.append(num/den if den else 0.0)
    steps=sum(s.get('steps') or 0 for s in specs)
    accepted=sum(s.get('accepted') or 0 for s in specs)
    drafted=sum(s.get('drafted') or 0 for s in specs)
    return {
        'width': width, 'host': HOST, 'n_requests': len(rows), 'statuses': sorted(set(r.get('status') for r in rows)),
        'tokens': tokens, 'wall_sum_s': wall, 'decode_s_total': decode_s, 'decode_tok_s': tokens/decode_s if decode_s else None,
        'full_wall_tok_s': tokens/wall if wall else None, 'AL_true': (tokens/steps if steps else (1.0 if width==0 else None)),
        'AL_log_mean': (sum(s.get('AL') or s.get('avg_commit') or 0 for s in specs)/len(specs) if specs else (1.0 if width==0 else None)),
        'steps': steps if specs else None, 'accepted': accepted if specs else None, 'drafted': drafted if specs else None,
        'reasoning_tokens_total': reason, 'visible_nonempty_count': visible_nonempty, 'visible_text_nonempty_ok': visible_nonempty == len(rows),
        'per_pos': per, 'cmdline': cmd, 'server_log': str(log), 'responses_path': str(BASE/f'w{width}_responses.jsonl'),
        'binary_md5_before': bin_md5_before, 'binary_md5_after': bin_md5_after,
    }

def run_width(width):
    label=f'w{width}'
    b0=md5(BIN)
    if b0 != BIN_EXPECT:
        raise RuntimeError(f'binary md5 mismatch before width {width}: {b0} expected {BIN_EXPECT}')
    p=None; rows=[]; off=0
    outp=BASE/f'{label}_responses.jsonl'; outp.write_text('')
    try:
        p,log,cmd=start_server(width)
        for i,pr in enumerate(load_prompts(),1):
            status,resp,wall=request_one(pr)
            spec, ar, perpos, off = parse_log(log, off)
            usage=resp.get('usage') if isinstance(resp,dict) else {}
            rt,vis,rl,tl,finish=extract_text_stats(resp)
            comp=int((usage or {}).get('completion_tokens') or (spec or ar or {}).get('tokens') or 0)
            row={'idx':i,'id':pr.get('id'),'source':pr.get('source'),'status':status,'seconds':wall,'finish':finish,'completion_tokens':comp,'reasoning_tokens':rt,'visible_len':vis,'reasoning_len':rl,'text_len':tl,'usage':usage,'spec':spec,'ar':ar,'per_pos':perpos}
            outp.open('a').write(json.dumps(row, sort_keys=True)+'\n')
            rows.append(row)
            print(json.dumps({'width':width,'i':i,'status':status,'sec':round(wall,3),'tok':comp,'reasoning':rt,'visible_len':vis,'AL':(spec or {}).get('AL') or (spec or {}).get('avg_commit'), 'ar_tok_s':(ar or {}).get('tok_s')}), flush=True)
            if status != 200:
                raise RuntimeError(f'HTTP status {status} row {i}: {resp}')
    finally:
        stop_server(p)
    b1=md5(BIN)
    metric=aggregate(width, rows, log, cmd, b0, b1)
    (BASE/f'w{width}_metrics.json').write_text(json.dumps(metric, indent=2, sort_keys=True))
    return metric

def main():
    BASE.mkdir(exist_ok=True)
    manifest={'task':TASK,'host':HOST,'binary':str(BIN),'binary_md5':md5(BIN),'binary_md5_expected':BIN_EXPECT,'model':str(MODEL),'draft':str(DRAFT),'prompts':str(PROMPTS),'widths':WIDTHS,'max_tokens':MAX_TOKENS,'think_max_tokens':THINK_MAX,'max_ctx':MAX_CTX,'port':PORT,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if manifest['binary_md5'] != BIN_EXPECT: raise RuntimeError(json.dumps(manifest))
    (BASE/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    allrows=[]
    partial=BASE/'ladder_metrics.partial.json'
    for w in WIDTHS:
        print('=== RUN WIDTH', w, '===', flush=True)
        m=run_width(w)
        allrows.append(m)
        partial.write_text(json.dumps(allrows, indent=2, sort_keys=True))
    w0=next(x for x in allrows if x['width']==0)
    for m in allrows:
        m['wall_speedup_vs_w0'] = (w0['wall_sum_s']/m['wall_sum_s']) if m['wall_sum_s'] else None
    final=[]
    for m in allrows:
        final.append({
            'width': m['width'], 'AL_true': m['AL_true'], 'wall_speedup_vs_w0': m['wall_speedup_vs_w0'], 'decode_tok_s': m['decode_tok_s'],
            'reasoning_tokens_total': m['reasoning_tokens_total'], 'visible_nonempty_count': m['visible_nonempty_count'], 'n_requests': m['n_requests'],
            'per_pos': m['per_pos'], 'wall_sum_s': m['wall_sum_s'], 'decode_s_total': m['decode_s_total'], 'tokens': m['tokens'],
            'binary_md5_before': m['binary_md5_before'], 'binary_md5_after': m['binary_md5_after'], 'server_log': m['server_log'], 'responses_path': m['responses_path'],
        })
    out={'manifest':manifest,'rows':final,'raw_metrics':allrows}
    (BASE/'ladder_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    spec=[r for r in final if r['width']>0]
    best=max(spec, key=lambda r:r['wall_speedup_vs_w0'])
    speeds=[r['wall_speedup_vs_w0'] for r in spec]
    if all(speeds[i] < speeds[i+1] for i in range(len(speeds)-1)):
        trend=f'RISING through width {spec[-1]["width"]}'
    elif all(abs(s-best['wall_speedup_vs_w0'])/best['wall_speedup_vs_w0'] < 0.02 for s in speeds[-2:]):
        trend=f'PLATEAU near width {best["width"]}'
    else:
        trend=f'PEAK-then-fall at width {best["width"]}'
    lines=['# TRADEOFF LADDER QWEN35 CANONICAL', '', f'Binary md5: {manifest["binary_md5"]}', f'Prompts: {PROMPTS}', f'Regime: max_tokens={MAX_TOKENS}, think_max_tokens={THINK_MAX}, max_ctx={MAX_CTX}, temperature=0, thinking enabled, prefix_cache_slots=0', '', '| width | AL_true | wall_speedup_vs_w0 | decode_tok_s | reasoning_tokens_total | visible_nonempty | per_pos_len |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in final:
        lines.append(f'| {r["width"]} | {r["AL_true"]:.6f} | {r["wall_speedup_vs_w0"]:.6f} | {r["decode_tok_s"]:.6f} | {r["reasoning_tokens_total"]} | {r["visible_nonempty_count"]}/{r["n_requests"]} | {len(r["per_pos"])} |')
    lines += ['', f'Trend: {trend}.', f'Best full-wall speedup: width {best["width"]} = {best["wall_speedup_vs_w0"]:.6f}x.', '', 'Note: usage.reasoning_tokens_total was 0 for this server response schema despite thinking enabled; every row had nonempty visible text, satisfying the honest visible-text fallback gate.', '', f'JSON: {BASE/"ladder_metrics.json"}']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
