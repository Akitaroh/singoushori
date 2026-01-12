"""
音楽サンプリング検出システム - エントリーポイント

このファイルからGUIアプリケーションを起動する。
"""

import sys
import os

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import main


if __name__ == "__main__":
    main()
