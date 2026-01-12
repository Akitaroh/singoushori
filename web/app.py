"""
サンプリング検出システム - Webアプリケーション
Flask + HTML/CSS/JavaScriptでローカルホスト
"""

import os
import sys
import tempfile
import traceback
import wave
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Render環境でlibrosa/numbaのJITコンパイルが重く、OOM/timeoutの原因になりやすいので無効化
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_processing import SamplingDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大50MB

# アップロードフォルダ
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 検出器のインスタンス（遅延初期化）
detector = None

def get_detector():
    global detector
    if detector is None:
        # メモリ節約のため低いサンプリングレートを使用
        detector = SamplingDetector(sr=16000, n_fft=1024, hop_length=256)
    return detector


def _allowed_extension(filename: str) -> bool:
    # RenderなどのLinux環境ではmp3デコード（ffmpeg）が無いことが多く失敗しやすい。
    # まずは確実に扱えるwavのみ許可（必要なら将来拡張）。
    _, ext = os.path.splitext(filename.lower())
    return ext in {'.wav'}


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


@app.route('/health')
def health():
    """ヘルスチェック＋デバッグ情報"""
    import numpy as np
    import scipy
    try:
        import librosa
        librosa_version = librosa.__version__
    except Exception as e:
        librosa_version = f"error: {e}"
    
    return jsonify({
        'ok': True,
        'python': sys.version,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'librosa': librosa_version,
        'numba_disable_jit': os.environ.get('NUMBA_DISABLE_JIT'),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })


@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    return jsonify({'error': 'ファイルが大きすぎます（50MB以内にしてください）'}), 413


@app.errorhandler(Exception)
def handle_exception(e):
    # HTTPException（404など）もJSONで返して、フロントでパースできるようにする
    if isinstance(e, HTTPException):
        return jsonify({'error': e.description}), e.code
    return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """サンプリング検出API"""
    original_path = None
    sample_path = None
    
    try:
        # ファイルの取得
        if 'original' not in request.files or 'sample' not in request.files:
            return jsonify({'error': 'ファイルが選択されていません'}), 400

        original_file = request.files['original']
        sample_file = request.files['sample']

        if original_file.filename == '' or sample_file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400

        # 拡張子チェック（デプロイ環境の安定性優先）
        if not _allowed_extension(original_file.filename) or not _allowed_extension(sample_file.filename):
            return jsonify({'error': 'デプロイ版はWAVのみ対応です（mp3はサーバ側でデコードできない場合があります）。wavに変換して再試行してください。'}), 400

        # パラメータの取得
        threshold = float(request.form.get('threshold', 0.2))
        tempo_min = float(request.form.get('tempo_min', 1.5))
        tempo_max = float(request.form.get('tempo_max', 2.0))
        tempo_step = float(request.form.get('tempo_step', 0.05))

        # 一時ファイルとして保存（ファイル名はサニタイズ）
        original_name = secure_filename(original_file.filename)
        sample_name = secure_filename(sample_file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{original_name}')
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sample_{sample_name}')

        original_file.save(original_path)
        sample_file.save(sample_path)
        
        # ファイルサイズ確認（メモリ節約のため厳しめに）
        original_size = os.path.getsize(original_path)
        sample_size = os.path.getsize(sample_path)
        
        if original_size == 0 or sample_size == 0:
            return jsonify({'error': 'アップロードされたファイルが空です'}), 400
        
        # 合計50MB以上は拒否
        total_size_mb = (original_size + sample_size) / (1024 * 1024)
        if total_size_mb > 50:
            return jsonify({'error': f'ファイルサイズが大きすぎます（合計{total_size_mb:.1f}MB）。50MB以内にしてください。'}), 400

        # 長さ制限（原曲5分、サンプル1分まで許可）
        try:
            original_sec = _wav_duration_seconds(original_path)
            sample_sec = _wav_duration_seconds(sample_path)
        except Exception:
            original_sec = None
            sample_sec = None

        if original_sec is not None and original_sec > 300:
            return jsonify({'error': f'原曲WAVが長すぎます（{original_sec:.1f}秒）。5分以内にしてください。'}), 400
        if sample_sec is not None and sample_sec > 60:
            return jsonify({'error': f'サンプルWAVが長すぎます（{sample_sec:.1f}秒）。1分以内にしてください。'}), 400
        
        print(f"[detect] Processing files: original={original_size/1024:.1f}KB, sample={sample_size/1024:.1f}KB", file=sys.stderr)

        # 検出実行
        det = get_detector()
        result = det.detect(
            original_path,
            sample_path,
            threshold=threshold,
            tempo_range=(tempo_min, tempo_max),
            tempo_step=tempo_step
        )

        # 結果を整形
        response = {
            'detected': result['detected'],
            'matches': result['matches'],
            'best_match': result['best_match'],
            'pitch_shift': result['pitch_shift'],
            'tempo_ratio': result['tempo_ratio'],
            'similarity_curve': result['similarity_curve'].tolist(),
            'curve_stats': {
                'max': float(result['similarity_curve'].max()),
                'mean': float(result['similarity_curve'].mean()),
                'length': len(result['similarity_curve'])
            }
        }

        return jsonify(response)
        
    except Exception as e:
        # 詳細なエラー情報を返す
        error_detail = traceback.format_exc()
        print(f"Detection error: {error_detail}", file=sys.stderr)
        return jsonify({
            'error': f'{type(e).__name__}: {str(e)}',
            'detail': error_detail[:1000]  # 最初の1000文字だけ
        }), 500
        
    finally:
        # 一時ファイル削除（例外時もクリーンアップ）
        if original_path and os.path.exists(original_path):
            os.remove(original_path)
        if sample_path and os.path.exists(sample_path):
            os.remove(sample_path)


if __name__ == '__main__':
    print("=" * 60)
    print("サンプリング検出システム - Webアプリケーション")
    print("=" * 60)
    port = int(os.environ.get('PORT', 8080))
    print(f"\nブラウザで http://localhost:{port} にアクセスしてください\n")
    app.run(debug=True, host='0.0.0.0', port=port)
