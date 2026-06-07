import pickle, numpy as np, sys
p = sys.argv[1]
with open(p, "rb") as f:
    d = pickle.load(f)
r = np.asarray(d["rewards"]); m = np.asarray(d["masks"]); dn = np.asarray(d["dones"])
print(f"file={p}")
print(f"  N={len(r)}  rewards: sum={r.sum():.1f} >0frac={(r>0).mean():.4f} max={r.max():.1f}")
print(f"  masks: mean={m.mean():.4f} (zeros={int((m==0).sum())})  dones: sum={dn.sum():.0f}")
obs = d["observations"]
if isinstance(obs, dict):
    img = np.asarray(obs["image"]); st = np.asarray(obs["state"])
    print(f"  obs.image: shape={img.shape} dtype={img.dtype} mean={img.mean():.1f}")
    print(f"  obs.state: shape={st.shape} dtype={st.dtype}")
print(f"  actions: shape={np.asarray(d['actions']).shape}")
