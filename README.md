# モザイク君 (Mozaikukun)

OBS Studio 用リアルタイム顔/物体検出マスキングプラグイン。

AIが自動で顔や物体を検出し、モザイク・ぼかし等のエフェクトをリアルタイムに適用します。

## 機能

- **顔検出** (YuNet) / **物体検出** (EdgeYOLO, COCO 80クラス)
- **20+マスキングエフェクト**: ぼかし、ピクセレート、すりガラス、グリッチ、グレースケール、サーマル、セピア、ネガティブ、チェッカー柄、ストライプ柄、ドット柄、単色塗り、目隠しバー、画像スタンプ、OBSソースオーバーレイ、マスク出力、透過
- **SORT追跡**: カルマンフィルタ + ハンガリアンアルゴリズムによる安定トラッキング
- **マスク形状**: 矩形 / 楕円、反転 (背景マスク)、膨張調整
- **クロップ**: 検出領域を限定
- **ズーム/フォロー**: 検出対象を自動追跡

## 対応環境

- Windows x64
- OBS Studio 30.x 以降
- GPU推論: DirectML (推奨) / CPU

## インストール

1. [Releases](https://github.com/rumda500/mozaikukun/releases) からzipをダウンロード
2. zipを展開
3. 中の `obs-plugins` フォルダと `data` フォルダを OBS のインストールフォルダにコピー
   - 通常: `C:\Program Files\obs-studio\`
4. OBS を起動 → ソースのフィルターに「モザイク君」が追加される

## 使い方

1. ソース (映像キャプチャ等) を右クリック → フィルター
2. 「+」→「モザイク君」を追加
3. マスキングオプションでエフェクトを選択
4. 必要に応じて Advanced Settings でモデルや閾値を調整

## ビルド

```powershell
# Windows x64
cmake -B build_x64 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_x64 --config RelWithDebInfo
```

## ライセンス

GPL-2.0 (GNU General Public License v2.0)

本プラグインは [obs-detect](https://github.com/locaal-ai/obs-detect) (by locaal-ai / Roy Shilkrot) をベースに開発されています。

## クレジット

- [obs-detect](https://github.com/locaal-ai/obs-detect) — 元プロジェクト (GPL-2.0)
- [YuNet](https://github.com/ShiqiYu/libfacedetection) — 顔検出モデル
- [EdgeYOLO](https://github.com/LSH9832/edgeyolo) — 物体検出モデル
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 推論エンジン
- [munkres-cpp](https://github.com/Gluttton/munkres-cpp) — ハンガリアンアルゴリズム (GPL-2.0)
