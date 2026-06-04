#!/usr/bin/env python3
import json, pathlib, re, time, urllib.request, os, sys
PORT=int(os.environ.get('PORT','8097'))
MODEL=os.environ.get('MODEL','minimax-luce-golden-revert')
URL=f"http://127.0.0.1:{PORT}/v1/completions"
LOG=pathlib.Path(os.environ['LUCE_LOG'])
OUTDIR=pathlib.Path(os.environ.get('OUTDIR','/home/<USER>/t_77088062'))
TAG=os.environ.get('TAG','revert')
OUTDIR.mkdir(parents=True, exist_ok=True)
PROMPTS=[
 'train 60 miles 1.5 hours',
 'Write a concise Python function that returns the nth Fibonacci number.',
 'Explain why the sky appears blue in one paragraph.',
 'Q: If a rectangle has width 8 and height 13, what is its area? A:',
 'Complete the sentence: The fastest way to improve inference throughput is',
 'Translate to Spanish: The new compiler pass reduced latency without changing model outputs.',
 'In two sentences, compare depth-first search and breadth-first search for graph traversal.',
]
spec_re=re.compile(r"\[spec-decode\].*tokens=(\d+).*steps=(\d+) accepted=(\d+)/(\d+) \(([^)]+)%\) avg_commit=([0-9.]+)")
vec_re=re.compile(r"\[spec-decode\] per_pos_(argmax_match|prefix_accept) p1\.\.p(\d+):\s*(.*)")
probe_re=re.compile(r"\[(draftattn-probe|draftmask-probe|rope-probe)\].*")
def parse_new(start,end):
 txt=LOG.read_text(errors='replace')[start:end]
 spec=arg=pref=None; probes=[]
 for line in txt.splitlines():
  m=spec_re.search(line)
  if m: spec={'tokens':int(m.group(1)),'steps':int(m.group(2)),'accepted':int(m.group(3)),'draft_total':int(m.group(4)),'accept_percent':float(m.group(5)),'avg_commit':float(m.group(6)),'line':line}
  m=vec_re.search(line)
  if m:
   vals=[float(x) for x in m.group(3).split()]
   if m.group(1)=='argmax_match': arg=vals
   else: pref=vals
  if probe_re.search(line): probes.append(line)
 return {'spec':spec,'per_pos_argmax_match':arg,'per_pos_prefix_accept':pref,'probes':probes,'log_excerpt':txt[-5000:]}
def post(prompt):
 body=json.dumps({'model':MODEL,'prompt':prompt,'max_tokens':64,'temperature':0,'top_k':1}).encode()
 req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json'})
 t0=time.time(); raw=urllib.request.urlopen(req,timeout=300).read().decode(); return time.time()-t0,json.loads(raw)
results=[]
for i,p in enumerate(PROMPTS,1):
 start=LOG.stat().st_size
 seconds,raw=post(p)
 time.sleep(.5); end=LOG.stat().st_size
 tel=parse_new(start,end)
 choice=raw.get('choices',[{}])[0]
 text=choice.get('text') or choice.get('message',{}).get('content') or ''
 rec={'i':i,'prompt':p,'seconds':seconds,'raw':raw,'telemetry':tel,'text':text}
 results.append(rec)
 print(json.dumps({'i':i,'spec':tel['spec'],'argmax':tel['per_pos_argmax_match'],'prefix':tel['per_pos_prefix_accept'],'probes':tel['probes'][:6]}), flush=True)
specs=[r['telemetry']['spec'] for r in results if r['telemetry'].get('spec')]
if len(specs)!=len(PROMPTS):
 raise SystemExit(f'missing spec telemetry {len(specs)}/{len(PROMPTS)}')
steps=sum(s['steps'] for s in specs); tokens=sum(s['tokens'] for s in specs); accepted=sum(s['accepted'] for s in specs); draft_total=sum(s['draft_total'] for s in specs)
weighted_arg=[sum(r['telemetry']['per_pos_argmax_match'][j]*r['telemetry']['spec']['steps'] for r in results)/steps for j in range(7)]
weighted_pref=[sum(r['telemetry']['per_pos_prefix_accept'][j]*r['telemetry']['spec']['steps'] for r in results)/steps for j in range(7)]
summary={'tag':TAG,'model':MODEL,'n_prompts':len(results),'max_tokens':64,'temperature':0,'top_k':1,'total_tokens':tokens,'total_steps':steps,'AL_true_mean_commit':tokens/steps,'accepted':accepted,'draft_total':draft_total,'accept_rate':accepted/draft_total,'per_pos_argmax_match_weighted_p1_p7':weighted_arg,'per_pos_prefix_accept_weighted_p1_p7':weighted_pref,'per_prompt':[{'i':r['i'],'prompt':r['prompt'],'seconds':r['seconds'],'spec':r['telemetry']['spec'],'per_pos_argmax_match':r['telemetry']['per_pos_argmax_match'],'per_pos_prefix_accept':r['telemetry']['per_pos_prefix_accept'],'probes':r['telemetry'].get('probes',[]),'text_preview':r['text'][:200]} for r in results]}
(OUTDIR/f'raw_completions_{TAG}_7prompts.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
(OUTDIR/f'measure_{TAG}_7prompts_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print('SUMMARY '+json.dumps(summary,ensure_ascii=False))
