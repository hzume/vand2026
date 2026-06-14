# Codex Instructions

このリポジトリでは、`input/` のCSVを読み込み、各実験ごとに独立したコードで予測・評価・提出ファイル作成を行う。

## Data

入力データは `input/` 配下に配置する。

| File | Columns | Description |
| --- | --- | --- |
| `input/Train.csv` | `ID`, `input`, `output`, `subset` | 学習用データ。`input` が質問・入力文、`output` が正解応答。 |
| `input/Val.csv` | `ID`, `input`, `output`, `subset` | 検証用データ。列の意味は `Train.csv` と同じ。 |
| `input/Test.csv` | `ID`, `input`, `subset` | 推論対象データ。各 `input` に対する応答を生成する。 |
| `input/SampleSubmission.csv` | `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM` | 提出ファイルの形式サンプル。 |

提出ファイルは `SampleSubmission.csv` と同じ列構成にし、`ID` の順序も `SampleSubmission.csv` に合わせる。

## Directory Layout

フォルダ構造は次を基本とする。

```text
input/
output/
exp00/
exp01/
exp02/
...
```

- `input/`: 配布データ置き場。
- `output/`: 実験結果、予測ファイル、提出CSVなどの出力先。
- `exp{id}/`: 実験ごとの独立したコード置き場。

`exp{id}` の `id` は `00`, `01`, `02`, ... の2桁連番にする。新しい実験を行う場合は既存の `exp{id}` を上書きせず、新しいフォルダを作成する。

各 `exp{id}` は独立して実行できる構成にし、他の `exp{id}` 配下のコードには依存させない。
