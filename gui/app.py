"""
音楽サンプリング検出システム - GUI モジュール

Tkinterベースの簡易GUI。
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

# matplotlibのバックエンドをTkAggに設定（インポート前に設定必須）
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 信号処理モジュールをインポート
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_processing import SamplingDetector


class SamplingDetectorApp:
    """音楽サンプリング検出GUIアプリケーション"""
    
    def __init__(self, root: tk.Tk):
        """
        Parameters
        ----------
        root : tk.Tk
            Tkinterのルートウィンドウ
        """
        self.root = root
        self.root.title("Music Sampling Detector")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        self.root.configure(bg='#f0f0f0')
        
        # ファイルパスを保持する変数
        self.original_path = tk.StringVar()
        self.sample_path = tk.StringVar()
        
        # 閾値
        self.threshold = tk.DoubleVar(value=0.7)
        
        # 検出器のインスタンス
        self.detector = SamplingDetector()
        
        # GUIを構築
        self._create_widgets()
    
    def _create_widgets(self):
        """GUIウィジェットを作成"""
        bg_color = '#f0f0f0'
        
        # メインフレーム
        main_frame = tk.Frame(self.root, padx=10, pady=10, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === ファイル選択セクション ===
        file_frame = tk.LabelFrame(main_frame, text="File Selection", padx=10, pady=10, bg=bg_color)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 原曲ファイル
        row1 = tk.Frame(file_frame, bg=bg_color)
        row1.pack(fill=tk.X, pady=5)
        tk.Label(row1, text="Original File:", width=15, anchor='w', bg=bg_color).pack(side=tk.LEFT)
        tk.Entry(row1, textvariable=self.original_path, width=55).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Browse...", command=self._select_original).pack(side=tk.LEFT)
        
        # 対象区間ファイル
        row2 = tk.Frame(file_frame, bg=bg_color)
        row2.pack(fill=tk.X, pady=5)
        tk.Label(row2, text="Sample File:", width=15, anchor='w', bg=bg_color).pack(side=tk.LEFT)
        tk.Entry(row2, textvariable=self.sample_path, width=55).pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Browse...", command=self._select_sample).pack(side=tk.LEFT)
        
        # === パラメータ設定セクション ===
        param_frame = tk.LabelFrame(main_frame, text="Parameters", padx=10, pady=10, bg=bg_color)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        param_row = tk.Frame(param_frame, bg=bg_color)
        param_row.pack(fill=tk.X)
        
        tk.Label(param_row, text="Detection Threshold:", bg=bg_color).pack(side=tk.LEFT, padx=(0, 10))
        self.threshold_slider = tk.Scale(
            param_row, 
            from_=0.5, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.threshold,
            length=300,
            resolution=0.01,
            bg=bg_color,
            highlightthickness=0
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        
        # === 実行セクション ===
        exec_frame = tk.Frame(main_frame, bg=bg_color)
        exec_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detect_button = tk.Button(
            exec_frame, 
            text="Start Detection", 
            command=self._start_detection,
            width=15,
            height=2,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold')
        )
        self.detect_button.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(exec_frame, text="Ready", bg=bg_color, font=('Arial', 11))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # === 結果表示セクション ===
        result_frame = tk.LabelFrame(main_frame, text="Detection Results", padx=10, pady=10, bg=bg_color)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 結果テキスト
        text_frame = tk.Frame(result_frame, bg=bg_color)
        text_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(text_frame, height=8, width=80, font=('Courier', 11))
        self.result_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # グラフ表示エリア
        graph_frame = tk.Frame(result_frame, bg='white')
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlibのfigureを作成
        self.figure = Figure(figsize=(8, 3), dpi=100, facecolor='white')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time (sec)")
        self.ax.set_ylabel("Similarity")
        self.ax.set_title("Similarity Curve")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _select_original(self):
        """原曲ファイルを選択"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.WAV *.MP3"),
            ("WAV files", "*.wav *.WAV"),
            ("MP3 files", "*.mp3 *.MP3"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="Select Original Audio",
            filetypes=filetypes
        )
        if path:
            self.original_path.set(path)
    
    def _select_sample(self):
        """対象区間ファイルを選択"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.WAV *.MP3"),
            ("WAV files", "*.wav *.WAV"),
            ("MP3 files", "*.mp3 *.MP3"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="Select Sample Audio",
            filetypes=filetypes
        )
        if path:
            self.sample_path.set(path)
    
    def _start_detection(self):
        """検出処理を開始"""
        # ファイルパスの確認
        original = self.original_path.get()
        sample = self.sample_path.get()
        
        if not original:
            messagebox.showerror("Error", "Please select the original audio file.")
            return
        
        if not sample:
            messagebox.showerror("Error", "Please select the sample audio file.")
            return
        
        if not os.path.exists(original):
            messagebox.showerror("Error", f"Original file not found:\n{original}")
            return
        
        if not os.path.exists(sample):
            messagebox.showerror("Error", f"Sample file not found:\n{sample}")
            return
        
        # UIを無効化
        self.detect_button.config(state=tk.DISABLED, bg='#cccccc')
        self.status_label.config(text="Processing...")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Processing... Please wait.\n")
        self.root.update()
        
        # 別スレッドで検出処理を実行
        thread = threading.Thread(
            target=self._run_detection,
            args=(original, sample, self.threshold.get())
        )
        thread.daemon = True
        thread.start()
    
    def _run_detection(self, original_path: str, sample_path: str, threshold: float):
        """検出処理を実行（別スレッド）"""
        try:
            # 検出実行
            result = self.detector.detect(original_path, sample_path, threshold)
            
            # メインスレッドで結果を表示
            self.root.after(0, lambda: self._show_result(result))
            
        except Exception as e:
            # エラーメッセージを表示
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _show_result(self, result: dict):
        """検出結果を表示"""
        # UIを有効化
        self.detect_button.config(state=tk.NORMAL, bg='#4CAF50')
        self.status_label.config(text="Done!")
        
        # 結果テキストを構築
        text = "=" * 50 + "\n"
        text += "Detection Result\n"
        text += "=" * 50 + "\n\n"
        
        if result["detected"]:
            text += "[OK] Sampling detected!\n\n"
            
            # 最良マッチ
            best = result["best_match"]
            if best:
                text += "[Best Match]\n"
                text += f"  Time: {best['time']:.2f} sec\n"
                text += f"  Similarity: {best['similarity']:.4f}\n"
                text += f"  Pitch Shift: {result['pitch_shift']} semitones\n\n"
            
            # 全マッチ
            text += f"[All Matches (Top {len(result['matches'])})]\n"
            for i, match in enumerate(result["matches"], 1):
                text += f"  {i}. Time: {match['time']:.2f}s, Similarity: {match['similarity']:.4f}\n"
        else:
            text += "[NG] No sampling detected.\n"
            text += "  Try lowering the threshold.\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        
        # グラフを更新
        self._update_graph(result)
    
    def _update_graph(self, result: dict):
        """類似度曲線のグラフを更新"""
        self.ax.clear()
        
        similarity = result["similarity_curve"]
        
        # 時間軸を生成
        time_axis = np.arange(len(similarity)) * self.detector.hop_length / self.detector.sr
        
        # 類似度曲線をプロット
        self.ax.plot(time_axis, similarity, 'b-', linewidth=1, label='Similarity')
        
        # 閾値ラインを描画
        threshold = self.threshold.get()
        self.ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        # 検出されたピークをマーク
        if result["detected"]:
            for match in result["matches"]:
                self.ax.axvline(x=match["time"], color='g', linestyle=':', alpha=0.7)
                self.ax.scatter([match["time"]], [match["similarity"]], 
                               color='g', s=100, zorder=5, marker='v')
        
        self.ax.set_xlabel("Time (sec)")
        self.ax.set_ylabel("Similarity")
        self.ax.set_title("Similarity Curve")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 1.1)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _show_error(self, message: str):
        """エラーを表示"""
        self.detect_button.config(state=tk.NORMAL, bg='#4CAF50')
        self.status_label.config(text="Error")
        
        messagebox.showerror("Error", f"Error during detection:\n{message}")
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Error: {message}")


def main():
    """GUIアプリケーションを起動"""
    root = tk.Tk()
    app = SamplingDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
