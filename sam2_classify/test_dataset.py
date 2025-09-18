import pandas as pd
from pathlib import Path

#这是prepared data的后一步，验证数据集划分train，val，test是否有重叠，检查toy那个数据集是否有缺失

root = Path("/home/wcheng31/sam2_classify")
df_tr = pd.read_csv(root/"train_manifest.csv")
df_va = pd.read_csv(root/"val_manifest.csv")
df_te = pd.read_csv(root/"test_manifest.csv")

def count_overlap(a, b, keys):
    ma = a[keys].drop_duplicates()
    mb = b[keys].drop_duplicates()
    m = ma.merge(mb, on=keys, how="inner")
    return len(m), len(ma), len(mb)

# 1) 最严格：image_path + tool
print("overlap (image_path, tool)")
for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
    n, na, nb = count_overlap(X[0], X[1], ["image_path","tool"])
    print(f"{name}: {n} / ({na}, {nb})")

# 2) 如果上面没重叠，再看 clip 级别（同 clip 高相似也会漏）
print("\noverlap by clip_name")
for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
    n, na, nb = count_overlap(X[0], X[1], ["clip_name","tool"])
    print(f"{name}: {n} / ({na}, {nb})")

# 3) 看同一帧 id（如果保留了）
if "frame_abs_index" in df_tr.columns and "frame_abs_index" in df_te.columns:
    print("\noverlap by (clip_name, frame_abs_index, tool)")
    for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
        n, na, nb = count_overlap(X[0], X[1], ["clip_name","frame_abs_index","tool"])
        print(f"{name}: {n} / ({na}, {nb})")


# import os, pandas as pd
# root="/home/wcheng31/sam2_classify"
# for name in ["train_manifest.csv","val_manifest.csv","test_manifest.csv"]:
#     p=f"{root}/{name}"
#     if not os.path.exists(p): 
#         print(name, "not found"); 
#         continue
#     df=pd.read_csv(p)
#     bad = ~df["image_path"].apply(os.path.exists)
#     print(f"{name}: missing {bad.sum()} / {len(df)}")

import os, pandas as pd
root="/home/wcheng31/sam2_classify"
for name in ["train_manifest.csv","val_manifest.csv","test_manifest.csv"]:
    p=f"{root}/{name}"
    if not os.path.exists(p): 
        continue
    df=pd.read_csv(p)
    keep = df["image_path"].apply(os.path.exists)
    removed = (~keep).sum()
    if removed:
        df = df[keep].reset_index(drop=True)
        df.to_csv(p, index=False)
        print(f"cleaned {name}: removed {removed} missing rows, kept {len(df)}")
    else:
        print(f"{name}: no missing rows")

