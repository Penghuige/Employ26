from pathlib import Path
import pandas as pd

input_name = "智联招聘_广东省_202203_202506"
input_csv = Path(rf"../data/{input_name}.csv")
out_dir = Path(r"../../output/split_out")
out_dir.mkdir(parents=True, exist_ok=True)

rows_per_file = 500_00   # 你可以调大/调小
reader = pd.read_csv(input_csv, chunksize=rows_per_file, encoding="utf-8", low_memory=False)

for i, chunk in enumerate(reader, start=1):
    out_file = out_dir / f"{input_name}_part_{i:04d}.csv"
    chunk.to_csv(out_file, index=False)  # 每个分片都带表头，便于单独使用
    print(f"{out_file} have been saved")