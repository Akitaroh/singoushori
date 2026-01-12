# 音楽サンプリング検出システム - 実装仕様書

## 概要

音楽サンプリング（楽曲の一部引用）を検出するシステムを実装する。
信号処理の理論に基づき、原曲と対象区間の類似度を計算し、サンプリング箇所を特定する。

## ファイル構成

```
project/
├── core/
│   └── signal_processing.py  # 信号処理の理論実装（このファイルが評価対象）
├── gui/
│   └── app.py                # 簡易GUI（Tkinter）
├── main.py                   # エントリーポイント
└── requirements.txt
```

---

## signal_processing.py の実装仕様

このファイルに全ての信号処理ロジックを実装する。

### 必要なライブラリ

```python
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import librosa  # 音声ファイル読み込み用
```

### クラス構成

```python
class SamplingDetector:
    """音楽サンプリング検出クラス"""
    
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        """
        Parameters:
        -----------
        sr : int
            サンプリングレート（デフォルト: 22050 Hz）
        n_fft : int
            FFTの窓幅（デフォルト: 2048サンプル）
        hop_length : int
            ホップサイズ（デフォルト: 512サンプル）
        """
```

### 実装すべきメソッド

以下のメソッドを順番に実装すること。各メソッドは対応する数式を実装する。

---

#### 1. `load_audio(self, file_path) -> np.ndarray`

音声ファイルを読み込み、モノラル化・正規化した離散信号を返す。

**処理内容:**
- WAV/MP3ファイルを読み込む
- ステレオの場合はモノラル化（左右チャンネルの平均）
- 振幅を[-1, 1]に正規化
- 指定されたサンプリングレートにリサンプリング

**戻り値:** `np.ndarray` - 1次元の離散信号 x[n]

---

#### 2. `hann_window(self, N) -> np.ndarray`

ハン窓を生成する。

**数式:**
$$w[n] = 0.5 \left(1 - \cos\frac{2\pi n}{N-1}\right), \quad n = 0, 1, ..., N-1$$

**Parameters:**
- `N`: 窓幅（サンプル数）

**戻り値:** `np.ndarray` - 長さNのハン窓

---

#### 3. `stft(self, x) -> np.ndarray`

短時間フーリエ変換を実行する。

**数式:**
$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j\frac{2\pi kn}{N}}$$

**処理内容:**
1. 信号xをフレームに分割（フレーム長: n_fft, ホップサイズ: hop_length）
2. 各フレームにハン窓を適用
3. 各フレームにFFTを適用

**Parameters:**
- `x`: 入力信号（1次元配列）

**戻り値:** `np.ndarray` - 複素数のSTFT行列（shape: [周波数ビン数, フレーム数]）

---

#### 4. `power_spectrogram(self, X) -> np.ndarray`

パワースペクトログラムを計算する。

**数式:**
$$P[m, k] = |X[m, k]|^2$$

**Parameters:**
- `X`: STFT行列

**戻り値:** `np.ndarray` - パワースペクトログラム

---

#### 5. `hz_to_chroma(self, freq) -> int`

周波数を音階クラス（0-11）に変換する。

**数式:**
$$p = \left\lfloor 12 \cdot \log_2\left(\frac{f}{440}\right) + 9 \right\rfloor \mod 12$$

**Parameters:**
- `freq`: 周波数（Hz）

**戻り値:** `int` - 音階クラス（0=C, 1=C#, 2=D, ..., 11=B）

---

#### 6. `compute_chroma(self, P) -> np.ndarray`

パワースペクトログラムからクロマ特徴量を計算する。

**数式:**
$$C[m, p] = \sum_{k \in \mathcal{K}_p} P[m, k]$$

**処理内容:**
1. 各周波数ビンkに対応する周波数を計算: f_k = k * sr / n_fft
2. 各周波数を音階クラスに変換（hz_to_chromaを使用）
3. 同じ音階クラスに属する周波数ビンのパワーを合計

**Parameters:**
- `P`: パワースペクトログラム

**戻り値:** `np.ndarray` - クロマ特徴量（shape: [12, フレーム数]）

---

#### 7. `normalize_chroma(self, C) -> np.ndarray`

クロマ特徴量を正規化する。

**処理内容:**
- 各フレームのL2ノルムで正規化
- ゼロ除算を避けるため、小さな値（epsilon=1e-10）を加算

**Parameters:**
- `C`: クロマ特徴量

**戻り値:** `np.ndarray` - 正規化されたクロマ特徴量

---

#### 8. `cross_correlation_fft(self, Cx, Cy) -> np.ndarray`

FFTを用いて相互相関を高速計算する。

**数式:**
$$R_{xy} = \mathcal{F}^{-1}\left\{ X^*[k] \cdot Y[k] \right\}$$

**処理内容:**
1. Cx, Cyをそれぞれ1次元に展開（フレーム方向に結合）
2. FFTを適用
3. 複素共役との積を計算
4. 逆FFTで相互相関を得る

**注意:** 長さが異なる場合はゼロパディングで揃える

**Parameters:**
- `Cx`: 原曲のクロマ特徴量
- `Cy`: 対象区間のクロマ特徴量

**戻り値:** `np.ndarray` - 相互相関配列

---

#### 9. `circular_shift_correlation(self, Cx, Cy) -> tuple[np.ndarray, np.ndarray]`

循環シフトを考慮した相互相関を計算し、ピッチシフトに対応する。

**数式:**
$$\text{Similarity}[\tau] = \max_{s=0}^{11} \hat{R}_{xy}[\tau, s]$$

**処理内容:**
1. s = 0, 1, ..., 11 の各シフト量について:
   - Cyを音階方向（12次元）にsだけ循環シフト: Cy_shifted[p] = Cy[(p+s) mod 12]
   - CxとCy_shiftedの相互相関を計算
   - 正規化
2. 各τについて、12パターンの最大値を採用
3. 最大類似度を与えたシフト量sも記録

**Parameters:**
- `Cx`: 原曲のクロマ特徴量（shape: [12, frames_x]）
- `Cy`: 対象区間のクロマ特徴量（shape: [12, frames_y]）

**戻り値:** 
- `similarity`: 各時刻τにおける最大類似度（1次元配列）
- `best_shift`: 各時刻τにおける最適シフト量（1次元配列）

---

#### 10. `detect_peaks(self, similarity, threshold=0.7) -> list[dict]`

類似度曲線からピークを検出する。

**処理内容:**
1. 閾値以上の類似度を持つ点を抽出
2. 極大点（前後より値が大きい点）を検出
3. 上位N件（デフォルト5件）を返す

**Parameters:**
- `similarity`: 類似度配列
- `threshold`: 検出閾値（デフォルト: 0.7）

**戻り値:** `list[dict]` - 検出結果のリスト
```python
[
    {
        "frame": int,        # フレームインデックス
        "time": float,       # 秒単位の時刻
        "similarity": float  # 類似度スコア
    },
    ...
]
```

---

#### 11. `detect(self, original_path, sample_path, threshold=0.7) -> dict`

メインの検出メソッド。全処理を統合して実行する。

**処理フロー:**
1. 原曲と対象区間を読み込み（load_audio）
2. 両方にSTFTを適用（stft）
3. パワースペクトログラムを計算（power_spectrogram）
4. クロマ特徴量を抽出・正規化（compute_chroma, normalize_chroma）
5. 循環相互相関を計算（circular_shift_correlation）
6. ピーク検出（detect_peaks）
7. 結果を返す

**Parameters:**
- `original_path`: 原曲のファイルパス
- `sample_path`: 対象区間のファイルパス
- `threshold`: 検出閾値

**戻り値:** `dict`
```python
{
    "detected": bool,           # サンプリングが検出されたか
    "matches": list[dict],      # 検出結果リスト（detect_peaksの戻り値）
    "best_match": dict | None,  # 最も類似度の高いマッチ
    "pitch_shift": int,         # 推定ピッチシフト量（半音単位）
    "similarity_curve": np.ndarray  # 類似度曲線（可視化用）
}
```

---

## GUI (app.py) の仕様

簡易的なTkinterベースのGUI。

### 機能要件

1. **ファイル選択**
   - 原曲ファイル選択ボタン
   - 対象区間ファイル選択ボタン
   - 対応形式: WAV, MP3

2. **パラメータ設定**
   - 閾値スライダー（0.5〜1.0、デフォルト0.7）

3. **実行**
   - 「検出開始」ボタン
   - プログレスバー（処理中表示）

4. **結果表示**
   - 検出有無
   - 検出時刻（秒単位、複数候補）
   - 類似度スコア
   - 推定ピッチシフト量
   - 類似度曲線のグラフ（matplotlib埋め込み）

---

## requirements.txt

```
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
matplotlib>=3.4.0
```

---

## 実装上の注意点

### 1. 数値安定性
- ゼロ除算を避けるため、正規化時に epsilon (1e-10) を加算
- 対数計算時は負の値や0を適切に処理

### 2. パフォーマンス
- FFTによる高速相互相関を使用
- 長い楽曲の場合はメモリ使用量に注意

### 3. エッジケース
- 非常に短い対象区間（1秒未満）への対応
- 無音区間の処理

### 4. コードの可読性
- 各メソッドに対応する数式をdocstringに記載
- 処理の各ステップにコメントを付与

---

## テスト方法

### 単体テスト
各メソッドが正しく動作するか確認:
- `hann_window`: 窓関数の形状確認
- `hz_to_chroma`: 既知の周波数での音階変換確認（A4=440Hz → 9）
- `stft`: 正弦波入力で期待する周波数にピークが出るか確認

### 統合テスト
- 同一ファイルを入力 → 類似度≒1.0
- 無関係なファイルを入力 → 類似度が低い
- ピッチシフトしたファイル → 正しいシフト量を検出

---

## 実装順序の推奨

1. `hann_window` - 単純な数式実装
2. `load_audio` - librosaを使用
3. `stft` - DFTの拡張
4. `power_spectrogram` - 単純な計算
5. `hz_to_chroma` - 単純な数式実装
6. `compute_chroma` - 周波数→音階の集約
7. `normalize_chroma` - 正規化
8. `cross_correlation_fft` - FFT応用
9. `circular_shift_correlation` - ピッチシフト対応
10. `detect_peaks` - ピーク検出
11. `detect` - 統合
