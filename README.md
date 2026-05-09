# VAND 2026 PatchCore Baseline

MVTec AD 2 形式のデータに対する one-class / unsupervised anomaly segmentation の最小 PatchCore baseline です。正常画像だけでカテゴリ別の memory bank を作り、各テスト画像に対して pixel-level anomaly map を `float16` TIFF として出力します。

## Setup

依存関係は `pyproject.toml` に定義しています。`pip install` や `uv pip install` は使わず、次のコマンドで仮想環境を同期してください。

```bash
uv sync
```

実行は `.venv/bin/python` を使います。

## Configuration

設定は [configs/patchcore.toml](configs/patchcore.toml) にあります。CLI 引数は用意していません。

主な設定:

- `paths.data_root`: データセットルート。既定は `input/mvtec_ad_2`
- `data.categories`: 実行対象カテゴリ
- `model.backbone`: `wide_resnet50_2` または `resnet50`
- `model.image_size`: train/test/private で共通の resize サイズ
- `patchcore.max_memory_patches`: memory bank の最大パッチ数
- `debug.enabled`: `true` のときだけ smoke test 用の画像数制限を有効化
- `debug.limit_*`: `debug.enabled = true` のときの画像数制限

## Dataset Inspection

```bash
.venv/bin/python scripts/inspect_dataset.py
```

カテゴリ、画像サイズ、split ごとの件数、`test_public/bad` と mask の対応を確認します。

## Train

```bash
.venv/bin/python scripts/train_patchcore.py
```

`train/good` の正常画像のみを使ってカテゴリ別 memory bank を作成し、`artifacts/patchcore/<category>/memory_bank.pt` に保存します。`configs/patchcore.toml` の `data.include_validation_good = true` にすると `validation/good` も正常画像として追加できます。

## Evaluate on test_public

```bash
.venv/bin/python scripts/evaluate_patchcore.py
```

`test_public/good` はゼロ mask、`test_public/bad` は `ground_truth/bad/<stem>_mask.png` を使って pixel AUROC、best F1、best IoU を計算します。Anomaly map は `outputs/patchcore/test_public/<category>/<stem>.tiff`、metrics は `outputs/patchcore/test_public/metrics.json` に保存します。

## Predict Private Splits

```bash
.venv/bin/python scripts/predict_patchcore.py
```

`configs/patchcore.toml` の `data.private_splits` にある `test_private` / `test_private_mixed` に対して anomaly map を生成します。出力先は `submissions/patchcore/<split>/<category>/<stem>.tiff` です。

## Smoke Test

短時間で動作確認したい場合は [configs/patchcore.toml](configs/patchcore.toml) の debug を一時的に有効化してください。`debug.enabled = false` のとき、`limit_*` は無視されます。

```toml
[debug]
enabled = true
limit_train_images = 2
limit_eval_images = 2
limit_predict_images = 2
```

その後、`inspect_dataset.py`、`train_patchcore.py`、`evaluate_patchcore.py`、`predict_patchcore.py` の順に実行します。
