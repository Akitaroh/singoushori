"""
signal_processing.py の詳細デバッグテスト
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.signal_processing import SamplingDetector


def detailed_test(original_path, sample_path):
    """詳細なステップバイステップテスト"""
    print("=" * 60)
    print("Detailed Step-by-Step Analysis")
    print("=" * 60)
    
    detector = SamplingDetector(sr=22050, n_fft=2048, hop_length=512)
    
    print(f"\nOriginal: {original_path}")
    print(f"Sample: {sample_path}")
    print(f"Expected sample location: 48.7s - 55.5s in original")
    
    # 1. 音声読み込み
    print("\n--- Step 1: Load Audio ---")
    original = detector.load_audio(original_path)
    sample = detector.load_audio(sample_path)
    
    print(f"Original: {len(original)} samples = {len(original)/detector.sr:.2f}s")
    print(f"Sample: {len(sample)} samples = {len(sample)/detector.sr:.2f}s")
    
    # 2. STFT
    print("\n--- Step 2: STFT ---")
    X_orig = detector.stft(original)
    X_samp = detector.stft(sample)
    
    print(f"Original STFT: {X_orig.shape} (bins, frames)")
    print(f"Sample STFT: {X_samp.shape}")
    print(f"Frame duration: {detector.hop_length/detector.sr*1000:.1f}ms")
    print(f"Total frames in original: {X_orig.shape[1]} = {X_orig.shape[1]*detector.hop_length/detector.sr:.2f}s")
    
    # 期待されるフレーム位置
    expected_start_frame = int(48.7 * detector.sr / detector.hop_length)
    expected_end_frame = int(55.5 * detector.sr / detector.hop_length)
    print(f"\nExpected frame range: {expected_start_frame} - {expected_end_frame}")
    
    # 3. パワースペクトログラム
    print("\n--- Step 3: Power Spectrogram ---")
    P_orig = detector.power_spectrogram(X_orig)
    P_samp = detector.power_spectrogram(X_samp)
    print(f"Original Power: {P_orig.shape}")
    print(f"Sample Power: {P_samp.shape}")
    
    # 4. クロマ特徴量
    print("\n--- Step 4: Chroma Features ---")
    C_orig = detector.compute_chroma(P_orig)
    C_samp = detector.compute_chroma(P_samp)
    print(f"Original Chroma: {C_orig.shape}")
    print(f"Sample Chroma: {C_samp.shape}")
    
    # 5. 正規化
    print("\n--- Step 5: Normalize ---")
    C_orig_norm = detector.normalize_chroma(C_orig)
    C_samp_norm = detector.normalize_chroma(C_samp)
    
    # 6. スライディング窓で類似度を直接計算
    print("\n--- Step 6: Direct Sliding Window Correlation ---")
    
    frames_orig = C_orig_norm.shape[1]
    frames_samp = C_samp_norm.shape[1]
    
    # スライディング窓による直接比較
    similarities = []
    
    for start in range(frames_orig - frames_samp + 1):
        # 原曲の対応区間を抽出
        window = C_orig_norm[:, start:start + frames_samp]
        
        # コサイン類似度を計算
        dot_product = np.sum(window * C_samp_norm)
        norm_window = np.linalg.norm(window)
        norm_samp = np.linalg.norm(C_samp_norm)
        
        if norm_window > 0 and norm_samp > 0:
            similarity = dot_product / (norm_window * norm_samp)
        else:
            similarity = 0
        
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    # 結果
    best_frame = np.argmax(similarities)
    best_time = best_frame * detector.hop_length / detector.sr
    best_similarity = similarities[best_frame]
    
    print(f"\nSliding window results:")
    print(f"  Best match frame: {best_frame}")
    print(f"  Best match time: {best_time:.2f}s")
    print(f"  Best similarity: {best_similarity:.4f}")
    print(f"  Expected time: 48.7s")
    print(f"  Time error: {abs(best_time - 48.7):.2f}s")
    
    # 上位5件
    print(f"\nTop 5 matches:")
    top_indices = np.argsort(similarities)[::-1][:5]
    for i, idx in enumerate(top_indices):
        time = idx * detector.hop_length / detector.sr
        print(f"  {i+1}. Frame {idx}, Time: {time:.2f}s, Similarity: {similarities[idx]:.4f}")
    
    # 期待される位置の類似度を確認
    print(f"\nSimilarity at expected location (frame {expected_start_frame}, ~48.7s):")
    if expected_start_frame < len(similarities):
        print(f"  Similarity: {similarities[expected_start_frame]:.4f}")
    
    # 類似度の統計
    print(f"\nSimilarity statistics:")
    print(f"  Max: {similarities.max():.4f}")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std: {similarities.std():.4f}")
    
    # プロット用にCSV出力
    print("\n--- Exporting similarity curve ---")
    output_file = "/Users/akitarohmac/singoushori/similarity_curve.csv"
    with open(output_file, 'w') as f:
        f.write("frame,time_sec,similarity\n")
        for i, sim in enumerate(similarities):
            time = i * detector.hop_length / detector.sr
            f.write(f"{i},{time:.3f},{sim:.6f}\n")
    print(f"Saved to: {output_file}")
    
    return similarities


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        original_path = sys.argv[1]
        sample_path = sys.argv[2]
        detailed_test(original_path, sample_path)
    else:
        print("Usage: python test_detailed.py <original.mp3> <sample.mp3>")
