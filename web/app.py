"""
サンプリング検出システム - Webアプリケーション
Flask + HTML/CSS/JavaScriptでローカルホスト
"""

import os
import sys
import json
import tempfile
from flask import Flask, render_template, request, jsonify

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_processing import SamplingDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大50MB

# アップロードフォルダ
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 検出器のインスタンス
detector = SamplingDetector()


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """サンプリング検出API"""
    try:
        # ファイルの取得
        if 'original' not in request.files or 'sample' not in request.files:
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        original_file = request.files['original']
        sample_file = request.files['sample']
        
        if original_file.filename == '' or sample_file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        # パラメータの取得
        threshold = float(request.form.get('threshold', 0.2))
        tempo_min = float(request.form.get('tempo_min', 1.5))
        tempo_max = float(request.form.get('tempo_max', 2.0))
        tempo_step = float(request.form.get('tempo_step', 0.05))
        
        # 一時ファイルとして保存
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + original_file.filename)
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_' + sample_file.filename)
        
        original_file.save(original_path)
        sample_file.save(sample_path)
        
        # 検出実行
        result = detector.detect(
            original_path, 
            sample_path, 
            threshold=threshold,
            tempo_range=(tempo_min, tempo_max),
            tempo_step=tempo_step
        )
        
        # 一時ファイル削除
        os.remove(original_path)
        os.remove(sample_path)
        
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("サンプリング検出システム - Webアプリケーション")
    print("=" * 60)
    port = int(os.environ.get('PORT', 8080))
    print(f"\nブラウザで http://localhost:{port} にアクセスしてください\n")
    app.run(debug=True, host='0.0.0.0', port=port)
