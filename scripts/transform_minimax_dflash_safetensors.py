#!/usr/bin/env python3
import json, struct, os, sys, shutil
src=sys.argv[1]
out=sys.argv[2]
with open(src,"rb") as f:
    hlen=struct.unpack("<Q", f.read(8))[0]
    hdr=json.loads(f.read(hlen))
base=8+hlen
entries=[]  # (out_name,dtype,shape,src_name,start,end)
def add(out_name, src_name, shape=None, slice_bytes=None):
    info=hdr[src_name]
    dtype=info["dtype"]
    start,end=info["data_offsets"]
    if slice_bytes is not None:
        rel0, rel1, newshape = slice_bytes
        entries.append((out_name,dtype,newshape,src_name,base+start+rel0,base+start+rel1))
    else:
        entries.append((out_name,dtype,shape or info["shape"],src_name,base+start,base+end))
# model-level tensors
for k in ["fc.weight","hidden_norm.weight","norm.weight","lm_head.weight","d2t","t2d"]:
    add(k,k)
# layer tensors: upstream checkpoint has layernorm packed under q_proj/mlp and fused fc1_weight = gate+up
for i in range(6):
    p=f"layers.{i}."
    add(p+"input_layernorm.weight", p+"self_attn.q_proj.layer_norm_weight")
    add(p+"post_attention_layernorm.weight", p+"mlp.layer_norm_weight")
    for sub in ["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight","self_attn.q_norm.weight","self_attn.k_norm.weight"]:
        add(p+sub, p+sub)
    fc1=p+"mlp.fc1_weight"
    info=hdr[fc1]
    assert info["dtype"]=="BF16", info
    rows, cols = info["shape"]
    assert rows % 2 == 0
    half=rows//2
    rowbytes=cols*2
    add(p+"mlp.gate_proj.weight", fc1, slice_bytes=(0, half*rowbytes, [half, cols]))
    add(p+"mlp.up_proj.weight", fc1, slice_bytes=(half*rowbytes, rows*rowbytes, [half, cols]))
    add(p+"mlp.down_proj.weight", p+"mlp.fc2_weight")
# Build safetensors header. Keep insertion order.
ohdr={}
off=0
for name,dtype,shape,srcname,start,end in entries:
    n=end-start
    ohdr[name]={"dtype":dtype,"shape":shape,"data_offsets":[off,off+n]}
    off += n
hbytes=json.dumps(ohdr,separators=(",",":"),sort_keys=False).encode()
tmp=out+".tmp"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(tmp,"wb") as fo, open(src,"rb") as fi:
    fo.write(struct.pack("<Q", len(hbytes)))
    fo.write(hbytes)
    buf=1024*1024*16
    for name,dtype,shape,srcname,start,end in entries:
        fi.seek(start)
        rem=end-start
        while rem:
            b=fi.read(min(buf,rem))
            if not b: raise RuntimeError("short read")
            fo.write(b); rem-=len(b)
os.replace(tmp,out)
print("wrote",out,"entries",len(entries),"bytes",os.path.getsize(out))
print("first",entries[0][:3],"last",entries[-1][:3])
