"""
signal_processing.py の動作テスト

コマンドラインから信号処理の各メソッドをテストする
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.signal_processing import SamplingDetector


def test_basic_methods():
    """基本メソッドのテスト"""
    print("=" * 60)
    print("Signal Processing Module Test")
    print("=" * 60)
    
    detector = SamplingDetector(sr=22050, n_fft=2048, hop_length=512)
    print(f"\n[OK] SamplingDetector initialized")
    print(f"     sr={detector.sr}, n_fft={detector.n_fft}, hop_length={detector.hop_length}")
    
    # 1. ハン窓テスト
    print("\n--- Test 1: hann_window ---")
    window = detector.hann_window(2048)
    print(f"[OK] Hann window generated: shape={window.shape}")
    print(f"     First 5 values: {window[:5]}")
    print(f"     Max value: {window.max():.4f} (expected: 1.0)")
    print(f"     Min value: {window.min():.4f} (expected: 0.0)")
    
    # 2. hz_to_chroma テスト
    print("\n--- Test 2: hz_to_chroma ---")
    test_freqs = [440, 261.63, 293.66, 329.63]  # A4, C4, D4, E4
    expected = [9, 0, 2, 4]  # A, C, D, E
    chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for freq, exp in zip(test_freqs, expected):
        result = detector.hz_to_chroma(freq)
        status = "[OK]" if result == exp else "[NG]"
        print(f"{status} {freq} Hz -> {result} ({chroma_names[result]}), expected: {exp} ({chroma_names[exp]})")
    
    # 3. STFT テスト（合成正弦波）
    print("\n--- Test 3: STFT with sine wave ---")
    duration = 1.0  # 1秒
    freq = 440  # 440Hz (A4)
    t = np.arange(int(detector.sr * duration)) / detector.sr
    sine_wave = np.sin(2 * np.pi * freq * t)
    print(f"     Generated sine wave: {freq}Hz, {duration}s, {len(sine_wave)} samples")
    
    X = detector.stft(sine_wave)
    print(f"[OK] STFT computed: shape={X.shape}")
    print(f"     Frequency bins: {X.shape[0]}, Time frames: {X.shape[1]}")
    
    # 4. パワースペクトログラム テスト
    print("\n--- Test 4: Power Spectrogram ---")
    P = detector.power_spectrogram(X)
    print(f"[OK] Power spectrogram computed: shape={P.shape}")
    print(f"     Max power: {P.max():.4f}")
    
    # 5. クロマ特徴量 テスト
    print("\n--- Test 5: Chroma Features ---")
    C = detector.compute_chroma(P)
    print(f"[OK] Chroma computed: shape={C.shape}")
    print(f"     Expected shape: (12, {P.shape[1]})")
    
    # 440Hzの正弦波なので、A（9番目）が最大になるはず
    mean_chroma = C.mean(axis=1)
    max_chroma_idx = np.argmax(mean_chroma)
    print(f"     Dominant chroma: {max_chroma_idx} ({chroma_names[max_chroma_idx]})")
    print(f"     Expected: 9 (A) for 440Hz sine wave")
    
    # 6. 正規化 テスト
    print("\n--- Test 6: Normalize Chroma ---")
    C_norm = detector.normalize_chroma(C)
    print(f"[OK] Normalized chroma: shape={C_norm.shape}")
    
    # 各フレームのL2ノルムが1に近いか確認
    norms = np.linalg.norm(C_norm, axis=0)
    print(f"     L2 norms (should be ~1.0): min={norms.min():.4f}, max={norms.max():.4f}")
    
    # 7. 相互相関 テスト
    print("\n--- Test 7: Cross Correlation ---")
    # 同じ信号で相互相関を計算
    correlation = detector.cross_correlation_fft(C_norm, C_norm)
    print(f"[OK] Cross correlation computed: length={len(correlation)}")
    print(f"     Max correlation: {correlation.max():.4f}")
    
    # 8. 循環シフト相互相関 テスト
    print("\n--- Test 8: Circular Shift Correlation ---")
    similarity, best_shift = detector.circular_shift_correlation(C_norm, C_norm)
    print(f"[OK] Circular shift correlation computed")
    print(f"     Similarity length: {len(similarity)}")
    print(f"     Max similarity: {similarity.max():.4f} (expected: ~1.0 for same signal)")
    print(f"     Best shift at max: {best_shift[np.argmax(similarity)]}")
    
    # 9. ピーク検出 テスト
    print("\n--- Test 9: Peak Detection ---")
    peaks = detector.detect_peaks(similarity, threshold=0.5)
    print(f"[OK] Peak detection completed")
    print(f"     Number of peaks found: {len(peaks)}")
    if peaks:
        print(f"     Top peak: time={peaks[0]['time']:.2f}s, similarity={peaks[0]['similarity']:.4f}")
    
    print("\n" + "=" * 60)
    print("All basic tests completed!")
    print("=" * 60)
    
    return True


def test_with_audio_files(original_path, sample_path):
    """実際の音声ファイルでテスト"""
    print("\n" + "=" * 60)
    print("Testing with Audio Files")
    print("=" * 60)
    
    detector = SamplingDetector()
    
    print(f"\nOriginal: {original_path}")
    print(f"Sample: {sample_path}")
    
    # ファイル存在チェック
    if not os.path.exists(original_path):
        print(f"[ERROR] Original file not found: {original_path}")
        return False
    if not os.path.exists(sample_path):
        print(f"[ERROR] Sample file not found: {sample_path}")
        return False
    
    print("\n--- Loading audio files ---")
    try:
        original = detector.load_audio(original_path)
        print(f"[OK] Original loaded: {len(original)} samples ({len(original)/detector.sr:.2f}s)")
        
        sample = detector.load_audio(sample_path)
        print(f"[OK] Sample loaded: {len(sample)} samples ({len(sample)/detector.sr:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        return False
    
    print("\n--- Running detection ---")
    try:
        result = detector.detect(original_path, sample_path, threshold=0.2)
        print(f"[OK] Detection completed!")
        print(f"\n--- Results ---")
        print(f"Detected: {result['detected']}")
        print(f"Number of matches: {len(result['matches'])}")
        print(f"Pitch shift: {result['pitch_shift']} semitones")
        
        if result['best_match']:
            print(f"\nBest match:")
            print(f"  Time: {result['best_match']['time']:.2f} sec")
            print(f"  Similarity: {result['best_match']['similarity']:.4f}")
        
        if result['matches']:
            print(f"\nAll matches:")
            for i, match in enumerate(result['matches'], 1):
                print(f"  {i}. Time: {match['time']:.2f}s, Similarity: {match['similarity']:.4f}")
        
        print(f"\nSimilarity curve stats:")
        curve = result['similarity_curve']
        print(f"  Length: {len(curve)} frames")
        print(f"  Max: {curve.max():.4f}")
        print(f"  Mean: {curve.mean():.4f}")
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # 基本テストを実行
    test_basic_methods()
    
    # コマンドライン引数があれば音声ファイルでテスト
    if len(sys.argv) >= 3:
        original_path = sys.argv[1]
        sample_path = sys.argv[2]
        test_with_audio_files(original_path, sample_path)
    else:
        print("\n" + "-" * 60)
        print("To test with audio files, run:")
        print("  python test_signal.py <original.wav> <sample.wav>")
        print("-" * 60)
