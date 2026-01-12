"""
音楽サンプリング検出システム - 信号処理モジュール

このファイルに全ての信号処理ロジックを実装する。
大学の信号処理課題の評価対象ファイル。
"""

import os
import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import resample


def _load_wav_scipy(file_path: str, target_sr: int) -> np.ndarray:
    """
    WAVファイルをscipy.io.wavfileで読み込む（librosa/numba依存を回避）。
    
    Parameters
    ----------
    file_path : str
        WAVファイルのパス
    target_sr : int
        目標サンプリングレート
    
    Returns
    -------
    np.ndarray
        モノラル化・正規化・リサンプリング済みの信号
    """
    sr_orig, data = wavfile.read(file_path)
    
    # int16/int32 → float64 に変換して正規化
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(np.float64)
    else:
        # uint8など
        data = data.astype(np.float64) / 255.0 - 0.5
    
    # ステレオ → モノラル
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    
    # リサンプリング
    if sr_orig != target_sr:
        num_samples = int(len(data) * target_sr / sr_orig)
        data = resample(data, num_samples)
    
    # 正規化
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    
    return data


# librosaはmp3等の読み込みに必要だが、Render等でnumba互換性問題がある場合はimportを遅延
_librosa = None

def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa as lb
        _librosa = lb
    return _librosa


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
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.wav':
            # WAVはscipy.io.wavfileで読み込む（numba依存を回避）
            return _load_wav_scipy(file_path, self.sr)
        else:
            # MP3等はlibrosaを使用（numbaが必要）
            librosa = _get_librosa()
            x, _ = librosa.load(file_path, sr=self.sr, mono=True)
            
            # 念のため正規化を確認
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
    
    def compute_chroma(self, P: np.ndarray, freq_min: float = 80.0, freq_max: float = 4000.0) -> np.ndarray:
        """
        パワースペクトログラムからクロマ特徴量を計算する。
        
        数式:
        C[m, p] = Σ_{k ∈ K_p} P[m, k]
        
        処理内容:
        1. 正の周波数ビンのみ使用（k = 1 ~ n_fft//2）
           ※ FFTの出力は対称なので、後半は負の周波数（ミラー）
        2. 帯域制限（デフォルト: 80Hz〜4000Hz）
           ※ DC成分や超低域・超高域のノイズを除外
        3. 各周波数ビンkに対応する周波数を計算: f_k = k * sr / n_fft
        4. 各周波数を音階クラスに変換（hz_to_chromaを使用）
        5. 同じ音階クラスに属する周波数ビンのパワーを合計
        
        Parameters
        ----------
        P : np.ndarray
            パワースペクトログラム（shape: [周波数ビン数, フレーム数]）
        freq_min : float
            使用する最低周波数（デフォルト: 80Hz、音楽の基音域下限）
        freq_max : float
            使用する最高周波数（デフォルト: 4000Hz、倍音の主要帯域上限）
        
        Returns
        -------
        np.ndarray
            クロマ特徴量（shape: [12, フレーム数]）
        """
        num_bins, num_frames = P.shape
        
        # クロマ特徴量を初期化（12音階クラス x フレーム数）
        C = np.zeros((12, num_frames))
        
        # 正の周波数ビンのみ使用（k = 1 ~ n_fft//2）
        # k=0 は DC成分（0Hz）なのでスキップ
        # k > n_fft//2 は負の周波数（FFTの対称性）なのでスキップ
        max_bin = min(num_bins, self.n_fft // 2 + 1)
        
        for k in range(1, max_bin):  # k=0（DC）をスキップ
            # 周波数ビンkに対応する周波数を計算
            # f_k = k * sr / n_fft
            freq = k * self.sr / self.n_fft
            
            # 帯域制限：音楽のピッチに関係ない周波数をスキップ
            if freq < freq_min or freq > freq_max:
                continue
            
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
    
    def _compute_single_tempo_curve(self, Cx: np.ndarray, Cy: np.ndarray, tempo_ratio: float) -> np.ndarray:
        """
        単一のテンポ倍率での類似度曲線を計算する。
        X軸が原曲の時間軸に対応する曲線を返す。
        
        Parameters
        ----------
        Cx : np.ndarray
            原曲のクロマ特徴量（shape: [12, frames_x]）
        Cy : np.ndarray
            サンプルのクロマ特徴量（shape: [12, frames_y]）
        tempo_ratio : float
            テンポ倍率
        
        Returns
        -------
        np.ndarray
            類似度曲線（長さ = 原曲のフレーム数 - リサンプル後サンプルのフレーム数 + 1）
        """
        # サンプルをテンポ倍率でリサンプリング
        Cy_resampled = self.resample_chroma(Cy, tempo_ratio)
        frames_x = Cx.shape[1]
        frames_y = Cy_resampled.shape[1]
        
        # サンプルが原曲より長い場合
        if frames_y > frames_x:
            return np.array([0.0])
        
        # スライディング窓の数
        num_positions = frames_x - frames_y + 1
        similarity = np.zeros(num_positions)
        
        # サンプルのL2ノルムを事前計算
        norm_Cy = np.linalg.norm(Cy_resampled)
        
        # 各スライド位置について処理
        for tau in range(num_positions):
            # 原曲の対応する窓を抽出
            window = Cx[:, tau:tau + frames_y]
            norm_window = np.linalg.norm(window)
            
            # ゼロ除算を回避
            if norm_window < 1e-10 or norm_Cy < 1e-10:
                similarity[tau] = 0.0
                continue
            
            # 12通りのピッチシフトで最大類似度を探索
            max_sim = -1.0
            for s in range(12):
                # Cyを音階方向にsだけ循環シフト
                Cy_shifted = np.roll(Cy_resampled, shift=s, axis=0)
                
                # コサイン類似度を計算
                dot_product = np.sum(window * Cy_shifted)
                cos_sim = dot_product / (norm_window * norm_Cy)
                
                if cos_sim > max_sim:
                    max_sim = cos_sim
            
            similarity[tau] = max_sim
        
        return similarity
    
    def resample_chroma(self, C: np.ndarray, tempo_ratio: float) -> np.ndarray:
        """
        クロマ特徴量を時間方向にリサンプリングする（テンポ変更対応）。
        
        Parameters
        ----------
        C : np.ndarray
            クロマ特徴量（shape: [12, frames]）
        tempo_ratio : float
            テンポ倍率（1.0 = 等速、2.0 = 2倍速）
        
        Returns
        -------
        np.ndarray
            リサンプリングされたクロマ特徴量
        """
        if tempo_ratio == 1.0:
            return C
        
        original_frames = C.shape[1]
        # テンポが速くなる = フレーム数が減る
        new_frames = int(original_frames / tempo_ratio)
        
        if new_frames < 1:
            new_frames = 1
        
        # 線形補間でリサンプリング
        original_indices = np.arange(original_frames)
        new_indices = np.linspace(0, original_frames - 1, new_frames)
        
        C_resampled = np.zeros((12, new_frames))
        for i in range(12):
            C_resampled[i, :] = np.interp(new_indices, original_indices, C[i, :])
        
        return C_resampled
    
    def circular_shift_correlation(self, Cx: np.ndarray, Cy: np.ndarray, tempo_ratios: list = None) -> tuple:
        """
        循環シフトを考慮した相互相関を計算し、ピッチシフトとテンポ変更に対応する。
        
        スライディング窓方式でコサイン類似度を計算する。
        
        数式:
        Similarity[τ] = max_{s=0}^{11} max_{r} cos_sim(Cx[τ:τ+L_r], shift(Cy_r, s))
        
        cos_sim(A, B) = (A · B) / (||A|| × ||B||)
        
        処理内容:
        1. 各テンポ倍率でサンプルをリサンプリング
        2. 原曲のクロマ特徴量に沿ってサンプルをスライド
        3. 各位置で、12通りのピッチシフト（循環シフト）を試行
        4. 各位置における最大類似度とそのときのシフト量・テンポ倍率を記録
        
        Parameters
        ----------
        Cx : np.ndarray
            原曲のクロマ特徴量（shape: [12, frames_x]）
        Cy : np.ndarray
            対象区間のクロマ特徴量（shape: [12, frames_y]）
        tempo_ratios : list
            テスト対象のテンポ倍率のリスト（デフォルト: [1.0]）
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            similarity : 各時刻τにおける最大類似度（1次元配列）
            best_shift : 各時刻τにおける最適シフト量（1次元配列）
            best_tempo : 各時刻τにおける最適テンポ倍率（1次元配列）
        """
        if tempo_ratios is None:
            tempo_ratios = [1.0]
        
        frames_x = Cx.shape[1]
        
        # 全テンポ倍率での結果を格納
        all_results = []
        
        for tempo_ratio in tempo_ratios:
            # サンプルをテンポ倍率でリサンプリング
            Cy_resampled = self.resample_chroma(Cy, tempo_ratio)
            frames_y = Cy_resampled.shape[1]
            
            # サンプルが原曲より長い場合はスキップ
            if frames_y > frames_x:
                continue
            
            # スライディング窓の数
            num_positions = frames_x - frames_y + 1
            
            # サンプルのL2ノルムを事前計算
            norm_Cy = np.linalg.norm(Cy_resampled)
            
            # 各スライド位置について処理
            for tau in range(num_positions):
                # 原曲の対応する窓を抽出
                window = Cx[:, tau:tau + frames_y]
                norm_window = np.linalg.norm(window)
                
                # ゼロ除算を回避
                if norm_window < 1e-10 or norm_Cy < 1e-10:
                    continue
                
                # 12通りのピッチシフトで最大類似度を探索
                max_sim = -1.0
                max_shift = 0
                
                for s in range(12):
                    # Cyを音階方向にsだけ循環シフト
                    Cy_shifted = np.roll(Cy_resampled, shift=s, axis=0)
                    
                    # コサイン類似度を計算
                    dot_product = np.sum(window * Cy_shifted)
                    cos_sim = dot_product / (norm_window * norm_Cy)
                    
                    if cos_sim > max_sim:
                        max_sim = cos_sim
                        max_shift = s
                
                all_results.append({
                    'tau': tau,
                    'similarity': max_sim,
                    'shift': max_shift,
                    'tempo_ratio': tempo_ratio
                })
        
        # 結果がない場合
        if not all_results:
            return np.array([0.0]), np.array([0]), np.array([1.0])
        
        # 全位置での最大類似度を持つ結果を構築
        # tau毎に最も良い結果を選択
        tau_best = {}
        for r in all_results:
            tau = r['tau']
            if tau not in tau_best or r['similarity'] > tau_best[tau]['similarity']:
                tau_best[tau] = r
        
        # ソートして配列に変換
        sorted_taus = sorted(tau_best.keys())
        similarity = np.array([tau_best[t]['similarity'] for t in sorted_taus])
        best_shift = np.array([tau_best[t]['shift'] for t in sorted_taus])
        best_tempo = np.array([tau_best[t]['tempo_ratio'] for t in sorted_taus])
        
        return similarity, best_shift, best_tempo
    
    def detect_peaks(self, similarity: np.ndarray, threshold: float = 0.7, 
                      best_tempo: np.ndarray = None, best_shift: np.ndarray = None,
                      top_n: int = 20) -> list:
        """
        類似度曲線からピークを検出する。
        
        処理内容:
        1. 閾値以上の類似度を持つ点を抽出
        2. 極大点（前後より値が大きい点）を検出
        3. 上位N件（デフォルト20件）を返す
        
        Parameters
        ----------
        similarity : np.ndarray
            類似度配列
        threshold : float
            検出閾値（デフォルト: 0.7）
        best_tempo : np.ndarray
            各位置での最適テンポ倍率（オプション）
        best_shift : np.ndarray
            各位置での最適ピッチシフト（オプション）
        top_n : int
            返す件数（デフォルト: 20）
        
        Returns
        -------
        list[dict]
            検出結果のリスト
            各要素は {"frame": int, "time": float, "similarity": float, "tempo_ratio": float, "pitch_shift": int}
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
                    
                    peak_data = {
                        "frame": i,
                        "time": time,
                        "similarity": float(similarity[i])
                    }
                    
                    # テンポ倍率を追加
                    if best_tempo is not None and i < len(best_tempo):
                        peak_data["tempo_ratio"] = float(best_tempo[i])
                    else:
                        peak_data["tempo_ratio"] = 1.0
                    
                    # ピッチシフトを追加（符号付き: -6〜+5 に変換）
                    if best_shift is not None and i < len(best_shift):
                        shift = int(best_shift[i])
                        # 0-11 を -6〜+5 に変換（例: 10 → -2, 2 → +2）
                        if shift > 6:
                            shift = shift - 12
                        peak_data["pitch_shift"] = shift
                    else:
                        peak_data["pitch_shift"] = 0
                    
                    peaks.append(peak_data)
        
        # 端点もチェック（最初と最後）
        if len(similarity) > 0:
            # 最初の点
            if similarity[0] >= threshold:
                if len(similarity) == 1 or similarity[0] > similarity[1]:
                    peak_data = {
                        "frame": 0,
                        "time": 0.0,
                        "similarity": float(similarity[0])
                    }
                    if best_tempo is not None and len(best_tempo) > 0:
                        peak_data["tempo_ratio"] = float(best_tempo[0])
                    else:
                        peak_data["tempo_ratio"] = 1.0
                    if best_shift is not None and len(best_shift) > 0:
                        shift = int(best_shift[0])
                        if shift > 6:
                            shift = shift - 12
                        peak_data["pitch_shift"] = shift
                    else:
                        peak_data["pitch_shift"] = 0
                    peaks.append(peak_data)
            
            # 最後の点
            if len(similarity) > 1 and similarity[-1] >= threshold:
                if similarity[-1] > similarity[-2]:
                    time = (len(similarity) - 1) * self.hop_length / self.sr
                    peak_data = {
                        "frame": len(similarity) - 1,
                        "time": time,
                        "similarity": float(similarity[-1])
                    }
                    if best_tempo is not None and len(best_tempo) > 0:
                        peak_data["tempo_ratio"] = float(best_tempo[-1])
                    else:
                        peak_data["tempo_ratio"] = 1.0
                    if best_shift is not None and len(best_shift) > 0:
                        shift = int(best_shift[-1])
                        if shift > 6:
                            shift = shift - 12
                        peak_data["pitch_shift"] = shift
                    else:
                        peak_data["pitch_shift"] = 0
                    peaks.append(peak_data)
        
        # 類似度の高い順にソートして上位N件を返す
        peaks.sort(key=lambda x: x["similarity"], reverse=True)
        
        return peaks[:top_n]
    
    def detect(self, original_path: str, sample_path: str, threshold: float = 0.7,
                tempo_range: tuple = (1.5, 2.0), tempo_step: float = 0.05) -> dict:
        """
        メインの検出メソッド。全処理を統合して実行する。
        テンポ変更（1倍〜3倍）にも対応。
        
        処理フロー:
        1. 原曲と対象区間を読み込み（load_audio）
        2. 両方にSTFTを適用（stft）
        3. パワースペクトログラムを計算（power_spectrogram）
        4. クロマ特徴量を抽出・正規化（compute_chroma, normalize_chroma）
        5. 複数テンポ倍率で循環相互相関を計算（circular_shift_correlation）
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
        tempo_range : tuple
            テンポ倍率の範囲（デフォルト: (1.5, 2.0)）
        tempo_step : float
            テンポ倍率の刻み幅（デフォルト: 0.05）
        
        Returns
        -------
        dict
            {
                "detected": bool,           # サンプリングが検出されたか
                "matches": list[dict],      # 検出結果リスト（各要素にtempo_ratio含む）
                "best_match": dict | None,  # 最も類似度の高いマッチ
                "pitch_shift": int,         # 推定ピッチシフト量（半音単位）
                "tempo_ratio": float,       # 推定テンポ倍率
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
        
        # 5. テンポ倍率のリストを生成（指定範囲を指定刻みで）
        num_steps = int((tempo_range[1] - tempo_range[0]) / tempo_step) + 1
        tempo_ratios = list(np.linspace(tempo_range[0], tempo_range[1], num_steps))
        
        # 循環相互相関を計算（複数テンポ対応）
        similarity, best_shift, best_tempo = self.circular_shift_correlation(
            C_original_norm, C_sample_norm, tempo_ratios
        )
        
        # 6. ピーク検出（テンポ・ピッチ情報付き、上位20件）
        matches = self.detect_peaks(similarity, threshold, best_tempo, best_shift, top_n=20)
        
        # 7. 結果を構築
        detected = len(matches) > 0
        best_match = matches[0] if detected else None
        
        # 最も高い類似度を持つ位置での情報を取得
        if best_match is not None:
            pitch_shift = best_match.get("pitch_shift", 0)
            # pitch_shiftは既にdetect_peaksで符号付きに変換済み
            tempo_ratio = best_match.get("tempo_ratio", 1.0)
        else:
            pitch_shift = 0
            tempo_ratio = 1.0
        
        # 原曲の長さ（秒）を計算
        original_duration_sec = len(original_signal) / self.sr
        
        # 最良テンポ倍率での類似度曲線を再計算（X軸を原曲の長さに合わせる）
        # 最良テンポでサンプルをリサンプリングし、スライド位置を原曲の時間軸に変換
        best_tempo_similarity = self._compute_single_tempo_curve(
            C_original_norm, C_sample_norm, tempo_ratio
        )
        
        # 時間軸を計算（各位置τを原曲内の秒数に変換）
        # τ番目の位置 = τ * hop_length / sr 秒
        num_positions = len(best_tempo_similarity)
        time_axis = np.array([i * self.hop_length / self.sr for i in range(num_positions)])
        
        return {
            "detected": detected,
            "matches": matches,
            "best_match": best_match,
            "pitch_shift": pitch_shift,
            "tempo_ratio": tempo_ratio,
            "similarity_curve": best_tempo_similarity,  # 最良テンポのみの曲線
            "time_axis": time_axis,  # 秒単位の時間軸
            "original_duration": original_duration_sec  # 原曲の長さ
        }
