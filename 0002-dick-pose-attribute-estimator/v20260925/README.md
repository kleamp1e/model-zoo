# 0002: dick-pose-attribute-estimator v20260925

画像1枚から陰茎のバウンディングボックス、姿勢2点、6ラベルの陽性確率を推定します。
姿勢推定はYOLO26n-pose、属性分類はEfficientNet-B0を使用します。
分類前の姿勢アライメントと正規化もONNXで実行します。

## 準備

`manifest.json`に記載されたURLから、ONNXファイルをダウンロードします。

```sh
curl -OL https://github.com/kleamp1e/model-zoo/releases/download/dick-pose-attribute-estimator-v20260925-rc1/kleamp1e-dick-pose-v20260925.onnx
curl -OL https://github.com/kleamp1e/model-zoo/releases/download/dick-pose-attribute-estimator-v20260925-rc1/kleamp1e-dick-attribute-preprocessor-v20260925.onnx
curl -OL https://github.com/kleamp1e/model-zoo/releases/download/dick-pose-attribute-estimator-v20260925-rc1/kleamp1e-dick-attribute-classifier-v20260925.onnx
```

依存ライブラリをインストールします。

```sh
uv sync --no-dev --locked
```

## 実行方法

```sh
uv run predict.py 画像ファイル
```

`--model-dir`でモデルと`manifest.json`があるディレクトリを変更できます。
既定値は`predict.py`のあるディレクトリです。
`--confidence`で姿勢検出の信頼度閾値を変更できます。
既定値は`0.25`で、NMSのIoU閾値は`0.7`です。

```sh
uv run predict.py --model-dir モデルディレクトリ --confidence 0.5 画像ファイル
```

標準出力には`{"detections": [...]}`形式のJSONを1行で出力します。
各要素の`box`は元画像座標の`[x1,y1,x2,y2]`、`confidence`は検出信頼度、`points`は元画像座標の`[[x,y],[x,y]]`、`attributes`は属性名をキーとする陽性確率です。
検出がない場合は`{"detections": []}`を返します。属性は独立したシグモイド確率なので、合計は1になりません。属性の判定閾値は提供していません。

## モデルの入出力

| モデル | 入力 | 出力 |
| --- | --- | --- |
| 姿勢推定 | `images`: `float32[1,3,640,640]`。RGB、レターボックス済み、0〜1 | `output0`: `float32[1,9,8400]`。候補ごとに中心座標、幅、高さ、信頼度、姿勢2点 |
| 前処理 | `image`: `uint8[H,W,3]`のRGB画像、`boxes`: `float32[N,4]`、`points`: `float32[N,2,2]` | `crops`: `float32[N,3,224,224]` |
| 属性分類 | `image`: `float32[N,3,224,224]` | `probabilities`: `float32[N,6]`。`sex`、`blowjob`、`handjob`、`boobjob`、`ejaculation`、`condom`の順 |

`predict.py`は信頼度フィルタ、NMS、元画像座標への変換を行った後、検出がある場合だけ前処理と分類を実行します。
前処理には姿勢推定と同じ向き・座標系のデコード済みRGB画像を渡します。

属性分類モデルの検証用205件におけるmacro PR-AUCは`0.739`です。
