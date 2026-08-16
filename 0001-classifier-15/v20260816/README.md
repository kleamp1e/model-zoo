# 0001: classifier-15 v20260816

## 準備

`manifest.json`に記載されたURLから、ONNXファイルをダウンロードします。

```sh
curl -OL https://github.com/kleamp1e/model-zoo/releases/download/classifier-15-v20260816-rc1/kleamp1e-classifier-15-preprocessor-v20260816.onnx
curl -OL https://github.com/kleamp1e/model-zoo/releases/download/classifier-15-v20260816-rc1/kleamp1e-classifier-15-v20260816.onnx
```

## 実行方法

`predict.py`は、画像1枚を推論して、15個の各ラベルの確率と、推奨閾値による判定結果を標準出力に出力します。

```sh
uv run predict.py 画像ファイル
```

モデルディレクトリ（`manifest.json`とONNXファイルが置かれたディレクトリ）は`--model-dir`で指定します。
省略した場合はカレントディレクトリを使用します。

```sh
uv run predict.py --model-dir モデルディレクトリ 画像ファイル
```

## 出力例

```
$ uv run predict.py sample.jpg
blowjob       probability=0.199405  threshold=0.931641  negative
censored      probability=0.016511  threshold=0.634277  negative
composite     probability=0.855583  threshold=0.398438  positive
cosplay       probability=0.042832  threshold=0.652344  negative
handjob       probability=0.138820  threshold=0.920898  negative
human         probability=0.011172  threshold=0.038116  negative
masturbation  probability=0.202600  threshold=0.939453  negative
nipple        probability=0.082361  threshold=0.670410  negative
photo         probability=0.102214  threshold=0.342285  negative
pornography   probability=0.011298  threshold=0.220459  negative
sex           probability=0.128553  threshold=0.323730  negative
sm            probability=0.102982  threshold=0.786133  negative
sperm         probability=0.067169  threshold=0.912598  negative
swimwear      probability=0.305715  threshold=0.884277  negative
underwear     probability=0.107862  threshold=0.527344  negative
```

各ラベルはシグモイドによる独立した確率です。確率の合計は1になりません。
判定は`probability >= threshold`で行っています。
