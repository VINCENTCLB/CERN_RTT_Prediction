"""Small Firefly supervised preprocessing and model smoke test."""
import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn


def quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def sample(con, path, splits, partition, count, seed):
    train_end = quoted(splits['train_end_inclusive'])
    val_end = quoted(splits['validation_end_inclusive'])
    condition = (f't <= TIMESTAMPTZ {train_end} AND e <= TIMESTAMPTZ {train_end}' if partition == 'train'
                 else f't > TIMESTAMPTZ {train_end} AND t <= TIMESTAMPTZ {val_end} AND e <= TIMESTAMPTZ {val_end}')
    query = f"""
        WITH f AS (
            SELECT *, TRY_CAST(flow_start AS TIMESTAMPTZ) AS t,
                   TRY_CAST(flow_end AS TIMESTAMPTZ) AS e
            FROM read_parquet({quoted(path)})
        )
        SELECT start_rtt, date_part('hour', t) AS hour_utc,
               date_part('dow', t) AS dow_utc,
               "meta.application" AS application,
               "meta.experiment_name" AS experiment,
               "meta.activity_id" AS activity,
               "meta.ip_version" AS ip_version,
               delta_rtt
        FROM f
        WHERE {condition} AND end_rtt <> 0
              AND isfinite(start_rtt) AND isfinite(delta_rtt)
        ORDER BY hash(flow_id, flow_start, {int(seed)}), flow_start, flow_id
        LIMIT {int(count)}
    """
    return con.execute(query).fetchdf()


def numeric(frame):
    rtt = frame['start_rtt'].to_numpy(dtype=float)
    hour = frame['hour_utc'].to_numpy(dtype=float)
    day = frame['dow_utc'].to_numpy(dtype=float)
    return np.column_stack((np.sign(rtt) * np.log1p(np.abs(rtt)),
                            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                            np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)))


def key(value):
    return None if pd.isna(value) else str(value)


def fit_transform(train, val):
    a, b = numeric(train), numeric(val)
    median = np.median(a, axis=0)
    scale = np.quantile(a, .75, axis=0) - np.quantile(a, .25, axis=0)
    scale[scale < 1e-8] = 1.0
    a, b = (a - median) / scale, (b - median) / scale
    train_parts, val_parts = [a], [b]
    for col in ['application', 'experiment', 'activity', 'ip_version']:
        counts = Counter(v for v in map(key, train[col]) if v is not None)
        values = sorted(counts, key=lambda v: (-counts[v], v))[:32]
        lookup = {v: i + 2 for i, v in enumerate(values)}
        def encode(frame):
            return np.asarray([0 if (v := key(x)) is None else lookup.get(v, 1)
                               for x in frame[col]], dtype=np.int64)
        train_ids, val_ids = encode(train), encode(val)
        card = len(values) + 2
        train_parts.append(np.eye(card, dtype=np.float32)[train_ids])
        val_parts.append(np.eye(card, dtype=np.float32)[val_ids])
    return (np.column_stack(train_parts).astype('float32'),
            np.column_stack(val_parts).astype('float32'))


def metrics(y, pred):
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-size', type=int, default=10000)
    p.add_argument('--validation-size', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output == args.dataset_dir or args.dataset_dir in args.output.parents:
        p.error('Choose an output file outside the original dataset directory.')
    if args.train_size < 2 or args.validation_size < 1 or args.epochs < 1:
        p.error('Use at least 2 training rows, 1 validation row and 1 epoch.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '2GB'")
    splits = json.loads((args.dataset_dir / 'splits.json').read_text())
    manifest = json.loads((args.dataset_dir / 'manifest.json').read_text())
    data = args.dataset_dir / 'flows.parquet'
    print('Selecting train sample...', flush=True)
    tr = sample(con, data, splits, 'train', args.train_size, args.seed)
    print('Selecting validation sample...', flush=True)
    va = sample(con, data, splits, 'validation', args.validation_size, args.seed)
    con.close()
    if len(tr) < 2 or len(va) < 1:
        raise ValueError('Too few eligible records for this pilot.')
    X, V = fit_transform(tr, va)
    y = tr['delta_rtt'].to_numpy(dtype=np.float64)
    vy = va['delta_rtt'].to_numpy(dtype=np.float64)
    center = float(np.median(y))
    scale = float(np.quantile(y, .75) - np.quantile(y, .25))
    if scale < 1e-8:
        scale = 1.0
    model = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.from_numpy(X)
    ty = torch.from_numpy(((y - center) / scale).astype('float32')).view(-1, 1)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        for ids in torch.randperm(len(X), generator=generator).split(256):
            opt.zero_grad()
            loss = loss_fn(model(tx[ids]), ty[ids])
            loss.backward()
            opt.step()
        print(f'Epoch {epoch+1}/{args.epochs} finished', flush=True)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(V)).flatten().numpy() * scale + center
    out = {'dataset_version': manifest['dataset_version'], 'sampling': 'DuckDB hash(flow_id, flow_start, seed), ordered ascending within eligible split',
           'seed': args.seed, 'train_rows': len(tr), 'validation_rows': len(va),
           'training_eligibility': 'start <= train cutoff and end <= train cutoff, end_rtt != 0',
           'validation_eligibility': 'train cutoff < start <= validation cutoff and end <= validation cutoff, end_rtt != 0',
           'inputs': ['start_rtt (signed log)', 'UTC hour sin/cos', 'UTC day sin/cos', 'application', 'experiment', 'activity', 'IP version'],
           'assumption': 'Predict after start_rtt and selected start context have arrived; verify availability and RTT units.',
           'target': 'delta_rtt, original source units',
           'models': {'zero_change': metrics(vy, np.zeros(len(vy))),
                      'training_median': metrics(vy, np.full(len(vy), center)),
                      'small_supervised_mlp': metrics(vy, pred)},
           'settings': {'hidden': 32, 'epochs': args.epochs, 'batch_size': 256, 'loss': 'SmoothL1 on train-median/IQR scaled target',
                        'torch': torch.__version__, 'duckdb': duckdb.__version__, 'numpy': np.__version__},
           'limitation': 'Pipeline smoke test only; not JEPA or a model comparison conclusion. The test partition was not queried.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
