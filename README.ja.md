# PixRestore for ComfyUI（日本語の概要）

[PixRestore](https://github.com/csslc/PixRestore) の公開1ステップモデル **PixRestore-S** を ComfyUI で使うための非公式ノードです。
ノイズ・ぼけ・JPEG 劣化を受けた **512x512 の RGB 画像**を、DINOv2 特徴を条件にしたデノイザー1回の呼び出しで復元します。

- 入出力は ComfyUI 標準の `IMAGE` です。ノードは Loader / Restore / Unload の3つです。
- 公式の1ステップ経路を**ビット単位で再現**しています。RTX 5090 の検証機では、未改変の公式 `inference.py` を eager モードで実行した結果と、
  8bit 出力・途中のテンソルがすべて一致しました（デモ12枚と center crop 4件、clean install でも同じ）。詳細は [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。
- 上流の `@torch.compile`（Triton が必要）をそのまま使う経路とは、出力の 8bit 値の 11% が最大 5/255、平均 0.11/255 ずれます。このノードは常に eager で動きます。
- このリポジトリには上流のコードも重みも含めていません。`tools/setup_pixrestore.py` が、固定リビジョンのファイルをダウンロードし、すべて SHA-256 で検証します。

> この復元は**生成的**です。劣化を取り除くと同時に、細部（質感・細い線）を作り出したり変えたりすることがあります。元の画像そのものを取り戻すものではありません。

## 導入

1. `ComfyUI/custom_nodes` に clone し、ComfyUI の Python で `pip install -r ComfyUI-PixRestore/requirements.txt` を実行します（timm）。
2. `python ComfyUI-PixRestore/tools/setup_pixrestore.py --models-dir ../models` を実行します（約 0.22 GB）。
3. ComfyUI を再起動し、`workflows/gui/pixrestore_restore.json` を開きます。試すときは `workflows/input` のサンプル画像を `ComfyUI/input` にコピーします。

## 使い方と制限

- `preprocess` は2種類です。
  - **exact 512x512**: 512x512 以外のサイズは拒否します。
  - **center crop to 512**: 公式の `--test-mode center_crop` と同じ処理です（短辺を bicubic で 512 にそろえ、中央 512x512 を使います）。
- 結果は `seed` によって変わります（既定は 0 で固定）。バッチは1枚ずつ `seed + index` で処理します（公式スクリプトでフォルダを処理するときと同じです）。
- 対応する範囲は、CUDA GPU のみ・PixRestore-S のみ・1ステップ・CFG 1.0 です。CPU、MPS、B/L/XL、タイル処理、512 以外の解像度での推論には対応していません。

## デモと実測

- デモは、作者自身の Looped-DiT 生成画像（合成画像）4枚に、3種類の劣化を加えた計12件です。全件を `docs/images` に置いています。
- PSNR は12件すべてで上がりました。ただし JPEG では +0.3〜0.8 dB にとどまります。
- これは少数の合成例であり、一般的な復元性能を示すベンチマークではありません。
- RTX 5090 での実測（ベンチマークではありません）
  - 初回: 3.7 秒（読み込みを含む）
  - 2回目以降: 1プロンプトあたり約 0.11 秒（うち GPU 処理は約 0.045 秒）
  - CUDA の最大割り当て: 342 MiB

## ライセンス

このリポジトリは MIT です。ダウンロードされるファイルの条件は [NOTICE](NOTICE) にまとめています。

- PixRestore の README は Apache 2.0 を宣言していますが、固定コミットには LICENSE ファイルがありません。一部のファイルには DiT などからの改変であることが記されています。
- モデルカードの表記は apache-2.0、DINOv2 は Apache 2.0 です。

利用目的に応じて、上流の条件を確認してください。
