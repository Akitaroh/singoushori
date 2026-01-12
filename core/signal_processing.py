"""
音楽サンプリング検出システム - 信号処理モジュール

このファイルに全ての信号処理ロジックを実装する。
大学の信号処理課題の評価対象ファイル。
"""

import numpy as np
from scipy.fft import fft, ifft
import librosa


class SamplingDetector:
    """
    音楽サンプリング検出クラス
    
    信号処理の理論に基づき、原曲と対象区間の類似度を計算し、
    サンプリング箇所を特定する。
    """
    
    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        """
        Parameters
        ----------
        sr : int
            サンプリングレート（デフォルト: 22050 Hz）
        n_fft : int
            FFTの窓幅（デフォルト: 2048サンプル）
        hop_length : int
            ホップサイズ（デフォルト: 512サンプル）
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        音声ファイルを読み込み、モノラル化・正規化した離散信号を返す。
        
        処理内容:
        - WAV/MP3ファイルを読み込む
        - ステレオの場合はモノラル化（左右チャンネルの平均）
        - 振幅を[-1, 1]に正規化
        - 指定されたサンプリングレートにリサンプリング
        
        Parameters
        ----------
        file_path : str
            音声ファイルのパス（WAV/MP3）
        
        Returns
        -------
        np.ndarray
            1次元の離散信号 x[n]
        """
        # librosaで音声ファイルを読み込み
        # mono=Trueでモノラル化、sr=self.srでリサンプリング
        # librosaは自動的に[-1, 1]に正規化する
        x, _ = librosa.load(file_path, sr=self.sr, mono=True)
        
        # 念のため正規化を確認（既にlibrosaで正規化されているが明示的に処理）
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val
        
        return x
    
    def hann_window(self, N: int) -> np.ndarray:
        """
        ハン窓を生成する。
        
        数式:
        w[n] = 0.5 * (1 - cos(2πn / (N-1))), n = 0, 1, ..., N-1
        
        Parameters
        ----------
        N : int
            窓幅（サンプル数）
        
        Returns
        -------
        np.ndarray
            長さNのハン窓
        """
        # n = 0, 1, ..., N-1 のインデックス配列を生成
        n = np.arange(N)
        
        # ハン窓の数式を実装
        # w[n] = 0.5 * (1 - cos(2πn / (N-1)))
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
        
        return w
    
    def stft(self, x: np.ndarray) -> np.ndarray:
        """
        短時間フーリエ変換（STFT）を実行する。
        
        数式:
        X[m, k] = Σ_{n=0}^{N-1} x[n + mH] · w[n] · e^{-j(2πkn/N)}
        
        処理内容:
        1. 信号xをフレームに分割（フレーム長: n_fft, ホップサイズ: hop_length）
        2. 各フレームにハン窓を適用
        3. 各フレームにFFTを適用
        
        Parameters
        ----------
        x : np.ndarray
            入力信号（1次元配列）
        
        Returns
        -------
        np.ndarray
            複素数のSTFT行列（shape: [周波数ビン数, フレーム数]）
        """
        N = self.n_fft      # FFT窓幅
        H = self.hop_length  # ホップサイズ
        
        # ハン窓を生成
        window = self.hann_window(N)
        
        # フレーム数を計算
        # 信号長がフレームに収まるようにパディング
        num_frames = 1 + (len(x) - N) // H
        if num_frames < 1:
            # 信号が短すぎる場合はゼロパディング
            x = np.pad(x, (0, N - len(x)))
            num_frames = 1
        
        # STFT行列を初期化（周波数ビン数 x フレーム数）
        # 周波数ビン数は N（FFTの出力サイズ）
        X = np.zeros((N, num_frames), dtype=np.complex128)
        
        # 各フレームについてFFTを計算
        for m in range(num_frames):
            # フレームの開始位置
            start = m * H
            
            # フレームを抽出（長さN）
            if start + N <= len(x):
                frame = x[start:start + N]
            else:
                # 信号の終端を超える場合はゼロパディング
                frame = np.zeros(N)
                frame[:len(x) - start] = x[start:]
            
            # ハン窓を適用
            windowed_frame = frame * window
            
            # FFTを適用
            # X[m, k] = Σ x[n + mH] · w[n] · e^{-j(2πkn/N)}
            X[:, m] = fft(windowed_frame)
        
        return X
    
    def power_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        パワースペクトログラムを計算する。
        
        数式:
        P[m, k] = |X[m, k]|^2
        
        Parameters
        ----------
        X : np.ndarray
            STFT行列
        
        Returns
        -------
        np.ndarray
            パワースペクトログラム
        """
        # 複素数の絶対値の2乗を計算
        P = np.abs(X) ** 2
        
        return P
    
    def hz_to_chroma(self, freq: float) -> int:
        """
        周波数を音階クラス（0-11）に変換する。
        
        数式:
        p = floor(12 · log2(f / 440) + 9) mod 12
        
        音階クラス: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
        
        Parameters
        ----------
        freq : float
            周波数（Hz）
        
        Returns
        -------
        int
            音階クラス（0-11）
        """
        # 周波数が0以下の場合はC（0）を返す
        if freq <= 0:
            return 0
        
        # 数式を実装
        # p = floor(12 · log2(f / 440) + 9) mod 12
        # A4 = 440Hz は音階クラス9（A）に対応
        p = int(np.floor(12 * np.log2(freq / 440) + 9)) % 12
        
        return p
    
    def compute_chroma(self, P: np.ndarray) -> np.ndarray:
        """
        パワースペクトログラムからクロマ特徴量を計算する。
        
        数式:
        C[m, p] = Σ_{k ∈ K_p} P[m, k]
        
        処理内容:
        1. 各周波数ビンkに対応する周波数を計算: f_k = k * sr / n_fft
        2. 各周波数を音階クラスに変換（hz_to_chromaを使用）
        3. 同じ音階クラスに属する周波数ビンのパワーを合計
        
        Parameters
        ----------
        P : np.ndarray
            パワースペクトログラム（shape: [周波数ビン数, フレーム数]）
        
        Returns
        -------
        np.ndarray
            クロマ特徴量（shape: [12, フレーム数]）
        """
        num_bins, num_frames = P.shape
        
        # クロマ特徴量を初期化（12音階クラス x フレーム数）
        C = np.zeros((12, num_frames))
        
        # 各周波数ビンについて処理
        for k in range(num_bins):
            # 周波数ビンkに対応する周波数を計算
            # f_k = k * sr / n_fft
            freq = k * self.sr / self.n_fft
            
            # 周波数を音階クラスに変換
            chroma_class = self.hz_to_chroma(freq)
            
            # 該当する音階クラスにパワーを加算
            # C[m, p] = Σ_{k ∈ K_p} P[m, k]
            C[chroma_class, :] += P[k, :]
        
        return C
    
    def normalize_chroma(self, C: np.ndarray) -> np.ndarray:
        """
        クロマ特徴量を正規化する。
        
        処理内容:
        - 各フレームのL2ノルムで正規化
        - ゼロ除算を避けるため、小さな値（epsilon=1e-10）を加算
        
        Parameters
        ----------
        C : np.ndarray
            クロマ特徴量（shape: [12, フレーム数]）
        
        Returns
        -------
        np.ndarray
            正規化されたクロマ特徴量
        """
        epsilon = 1e-10
        
        # 各フレーム（列）のL2ノルムを計算
        # axis=0 で各列（フレーム）ごとにノルムを計算
        l2_norm = np.linalg.norm(C, axis=0, keepdims=True)
        
        # ゼロ除算を避けるためepsilonを加算して正規化
        C_normalized = C / (l2_norm + epsilon)
        
        return C_normalized
    
    def cross_correlation_fft(self, Cx: np.ndarray, Cy: np.ndarray) -> np.ndarray:
        """
        FFTを用いて相互相関を高速計算する。
        
        数式:
        R_xy = F^{-1}{ X*[k] · Y[k] }
        
        処理内容:
        1. Cx, Cyをそれぞれ1次元に展開（フレーム方向に結合）
        2. FFTを適用
        3. 複素共役との積を計算
        4. 逆FFTで相互相関を得る
        
        注意: 長さが異なる場合はゼロパディングで揃える
        
        Parameters
        ----------
        Cx : np.ndarray
            原曲のクロマ特徴量
        Cy : np.ndarray
            対象区間のクロマ特徴量
        
        Returns
        -------
        np.ndarray
            相互相関配列
        """
        # 1次元に展開（フレーム方向に結合）
        cx_flat = Cx.flatten('F')  # Fortran順（列優先）で展開
        cy_flat = Cy.flatten('F')
        
        # 長さを揃えるためのゼロパディング
        # 相互相関のためには、長さを n + m - 1 にする必要がある
        n = len(cx_flat)
        m = len(cy_flat)
        fft_size = n + m - 1
        
        # 2のべき乗に丸めて高速化（オプション）
        fft_size = int(2 ** np.ceil(np.log2(fft_size)))
        
        # ゼロパディングしてFFTを適用
        Cx_fft = fft(cx_flat, n=fft_size)
        Cy_fft = fft(cy_flat, n=fft_size)
        
        # 複素共役との積を計算
        # R_xy = F^{-1}{ X*[k] · Y[k] }
        cross_spectrum = np.conj(Cx_fft) * Cy_fft
        
        # 逆FFTで相互相関を得る
        R_xy = ifft(cross_spectrum)
        
        # 実部のみを返す（虚部は数値誤差）
        return np.real(R_xy)
    
    def circular_shift_correlation(self, Cx: np.ndarray, Cy: np.ndarray) -> tuple:
        """
        循環シフトを考慮した相互相関を計算し、ピッチシフトに対応する。
        
        スライディング窓方式でコサイン類似度を計算する。
        
        数式:
        Similarity[τ] = max_{s=0}^{11} cos_sim(Cx[τ:τ+L], shift(Cy, s))
        
        cos_sim(A, B) = (A · B) / (||A|| × ||B||)
        
        処理内容:
        1. 原曲のクロマ特徴量に沿ってサンプルをスライド
        2. 各位置で、12通りのピッチシフト（循環シフト）を試行
        3. 各位置における最大類似度とそのときのシフト量を記録
        
        Parameters
        ----------
        Cx : np.ndarray
            原曲のクロマ特徴量（shape: [12, frames_x]）
        Cy : np.ndarray
            対象区間のクロマ特徴量（shape: [12, frames_y]）
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            similarity : 各時刻τにおける最大類似度（1次元配列、長さ = frames_x - frames_y + 1）
            best_shift : 各時刻τにおける最適シフト量（1次元配列）
        """
        frames_x = Cx.shape[1]
        frames_y = Cy.shape[1]
        
        # サンプルが原曲より長い場合はエラー回避
        if frames_y > frames_x:
            # 空の結果を返す
            return np.array([0.0]), np.array([0])
        
        # スライディング窓の数
        num_positions = frames_x - frames_y + 1
        
        # 結果を格納する配列
        similarity = np.zeros(num_positions)
        best_shift = np.zeros(num_positions, dtype=int)
        
        # サンプルのL2ノルムを事前計算
        norm_Cy = np.linalg.norm(Cy)
        
        # 各スライド位置について処理
        for tau in range(num_positions):
            # 原曲の対応する窓を抽出
            window = Cx[:, tau:tau + frames_y]
            norm_window = np.linalg.norm(window)
            
            # ゼロ除算を回避
            if norm_window < 1e-10 or norm_Cy < 1e-10:
                similarity[tau] = 0.0
                best_shift[tau] = 0
                continue
            
            # 12通りのピッチシフトで最大類似度を探索
            max_sim = -1.0
            max_shift = 0
            
            for s in range(12):
                # Cyを音階方向にsだけ循環シフト
                # Cy_shifted[p] = Cy[(p+s) mod 12]
                Cy_shifted = np.roll(Cy, shift=s, axis=0)
                
                # コサイン類似度を計算
                # cos_sim = (A · B) / (||A|| × ||B||)
                dot_product = np.sum(window * Cy_shifted)
                cos_sim = dot_product / (norm_window * norm_Cy)
                
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    max_shift = s
            
            similarity[tau] = max_sim
            best_shift[tau] = max_shift
        
        return similarity, best_shift
    
    def detect_peaks(self, similarity: np.ndarray, threshold: float = 0.7) -> list:
        """
        類似度曲線からピークを検出する。
        
        処理内容:
        1. 閾値以上の類似度を持つ点を抽出
        2. 極大点（前後より値が大きい点）を検出
        3. 上位N件（デフォルト5件）を返す
        
        Parameters
        ----------
        similarity : np.ndarray
            類似度配列
        threshold : float
            検出閾値（デフォルト: 0.7）
        
        Returns
        -------
        list[dict]
            検出結果のリスト
            各要素は {"frame": int, "time": float, "similarity": float}
        """
        peaks = []
        
        # 極大点の検出（前後より値が大きい点）
        for i in range(1, len(similarity) - 1):
            # 閾値以上かつ極大点であるかチェック
            if similarity[i] >= threshold:
                if similarity[i] > similarity[i - 1] and similarity[i] > similarity[i + 1]:
                    # フレームインデックスから時刻を計算
                    # time = frame * hop_length / sr
                    time = i * self.hop_length / self.sr
                    
                    peaks.append({
                        "frame": i,
                        "time": time,
                        "similarity": float(similarity[i])
                    })
        
        # 端点もチェック（最初と最後）
        if len(similarity) > 0:
            # 最初の点
            if similarity[0] >= threshold:
                if len(similarity) == 1 or similarity[0] > similarity[1]:
                    peaks.append({
                        "frame": 0,
                        "time": 0.0,
                        "similarity": float(similarity[0])
                    })
            
            # 最後の点
            if len(similarity) > 1 and similarity[-1] >= threshold:
                if similarity[-1] > similarity[-2]:
                    time = (len(similarity) - 1) * self.hop_length / self.sr
                    peaks.append({
                        "frame": len(similarity) - 1,
                        "time": time,
                        "similarity": float(similarity[-1])
                    })
        
        # 類似度の高い順にソートして上位5件を返す
        peaks.sort(key=lambda x: x["similarity"], reverse=True)
        
        return peaks[:5]
    
    def detect(self, original_path: str, sample_path: str, threshold: float = 0.7) -> dict:
        """
        メインの検出メソッド。全処理を統合して実行する。
        
        処理フロー:
        1. 原曲と対象区間を読み込み（load_audio）
        2. 両方にSTFTを適用（stft）
        3. パワースペクトログラムを計算（power_spectrogram）
        4. クロマ特徴量を抽出・正規化（compute_chroma, normalize_chroma）
        5. 循環相互相関を計算（circular_shift_correlation）
        6. ピーク検出（detect_peaks）
        7. 結果を返す
        
        Parameters
        ----------
        original_path : str
            原曲のファイルパス
        sample_path : str
            対象区間のファイルパス
        threshold : float
            検出閾値（デフォルト: 0.7）
        
        Returns
        -------
        dict
            {
                "detected": bool,           # サンプリングが検出されたか
                "matches": list[dict],      # 検出結果リスト
                "best_match": dict | None,  # 最も類似度の高いマッチ
                "pitch_shift": int,         # 推定ピッチシフト量（半音単位）
                "similarity_curve": np.ndarray  # 類似度曲線（可視化用）
            }
        """
        # 1. 原曲と対象区間を読み込み
        original_signal = self.load_audio(original_path)
        sample_signal = self.load_audio(sample_path)
        
        # 2. 両方にSTFTを適用
        X_original = self.stft(original_signal)
        X_sample = self.stft(sample_signal)
        
        # 3. パワースペクトログラムを計算
        P_original = self.power_spectrogram(X_original)
        P_sample = self.power_spectrogram(X_sample)
        
        # 4. クロマ特徴量を抽出・正規化
        C_original = self.compute_chroma(P_original)
        C_sample = self.compute_chroma(P_sample)
        
        C_original_norm = self.normalize_chroma(C_original)
        C_sample_norm = self.normalize_chroma(C_sample)
        
        # 5. 循環相互相関を計算
        similarity, best_shift = self.circular_shift_correlation(
            C_original_norm, C_sample_norm
        )
        
        # 6. ピーク検出
        matches = self.detect_peaks(similarity, threshold)
        
        # 7. 結果を構築
        detected = len(matches) > 0
        best_match = matches[0] if detected else None
        
        # 最も高い類似度を持つ位置でのピッチシフト量を取得
        if best_match is not None:
            pitch_shift = int(best_shift[best_match["frame"]])
        else:
            pitch_shift = 0
        
        return {
            "detected": detected,
            "matches": matches,
            "best_match": best_match,
            "pitch_shift": pitch_shift,
            "similarity_curve": similarity
        }
