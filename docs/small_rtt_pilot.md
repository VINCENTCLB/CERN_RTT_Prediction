# Small preprocessing and model pilot

This pilot uses a 10,000-flow training sample and a 2,000-flow validation sample. It tests a candidate start-time preprocessing pipeline and a small supervised MLP. Results are exploratory and do not demonstrate JEPA performance.

## Candidate choices

- Predict `delta_rtt` after the start RTT and selected start-event context arrive. Their online availability still needs confirmation.
- Parse UTC timestamps using the provided split boundaries. Exclude zero end RTT from supervised labels. For this pilot, a flow must also end by its own split's cutoff; end time is only a proxy for label delivery. Legitimate negative changes remain.
- Select samples by DuckDB hash of the reconstruction key and seed 42 within each eligible partition. The test partition is not evaluated.
- Use signed-log start RTT, cyclical UTC time, application, experiment, activity and IP version. Exclude CRIC, raw IP and ports from the first pilot; their inclusion remains a group decision.
- Fit numeric median/IQR scales and the top-32 category vocabularies on the sampled training data only. Missing and unseen categories have separate encodings. No end-event fields enter the model inputs.
- Compare zero change, the training median, and a one-hidden-layer MLP. The MLP uses SmoothL1 loss and an eight-epoch fixed run. MAE/RMSE use source RTT units, whose physical unit remains unverified.

## Run

Use Python 3.12 and install DuckDB, NumPy, pandas and PyTorch. The versions used here were DuckDB 2.0.0, NumPy 2.5.5, pandas 2.2.3 and PyTorch 2.15.0. From the repository root:

```bash
python -m pip install duckdb==2.0.0 numpy==2.5.5 pandas==2.2.3 torch==2.15.0
python processing/small_rtt_pilot.py --dataset-dir student_dataset --output data/small_rtt_pilot/results.json
python -m unittest discover -s tests -p 'test_small_rtt_pilot.py'
```

In a separate local check of the teammate's [`encoder.py`](https://github.com/VINCENTCLB/CERN_RTT_Prediction/blob/vincent/architecture/encoder.py), an indentation issue caused predictor methods to overwrite embedding methods. After a temporary local correction, the components completed one forward/backward pass on 16 sampled flows. That check is not part of this pilot script and did not modify the repository or perform JEPA training.

The supervised MLP achieved validation MAE 2.34 and RMSE 14.23; the zero-change baseline achieved MAE 4.26 and RMSE 14.72. Values are in uncalibrated source RTT units. These are small-sample results, not a final comparison. No raw records or samples are saved. The pilot does not implement a target encoder, EMA, JEPA pretraining or downstream JEPA evaluation.
