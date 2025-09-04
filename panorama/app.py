#!/usr/bin/env python3
"""
PANORAMA - Main application entry.
Initializes QApplication, applies PyQtDarkTheme, launches main window.
"""

import sys
from PyQt5.QtWidgets import QApplication
from panorama.main_rssi import PanoramaAppWindow

# Try import PyQtDarkTheme (qdarktheme)
try:
    import qdarktheme  # PyQtDarkTheme
except Exception:  # not installed or failed
    qdarktheme = None


def main():
    """Application main."""
    try:
        import os
        # Проверяем, нужно ли включить WebGL
        if os.environ.get("PANORAMA_WEBGL", "0") == "1":
            # ВКЛЮЧАЕМ поддержку WebGL и OpenGL
            os.environ.setdefault("QT_OPENGL", "desktop")
            os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "0")
            
            # Проверяем, есть ли дополнительные флаги
            if os.environ.get("PANORAMA_WEBGL_FORCE", "0") == "1":
                # Принудительное включение WebGL с дополнительными флагами
                os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--enable-gpu --enable-software-rasterizer --disable-web-security --disable-features=VizDisplayCompositor --disable-gpu-sandbox --disable-gpu-driver-bug-workarounds --ignore-gpu-blocklist --enable-unsafe-webgpu")
                print("🌐 WebGL поддержка включена (принудительно)")
            else:
                # Обычное включение WebGL
                os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--enable-gpu --enable-software-rasterizer --ignore-gpu-blocklist")
                print("🌐 WebGL поддержка включена")
            
            os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        else:
            # ОТКЛЮЧАЕМ для совместимости (по умолчанию)
            os.environ.setdefault("QT_OPENGL", "software")
            os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
            os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer --in-process-gpu")
            os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
            from PyQt5 import QtCore
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
    except Exception:
        pass

    # For PyQt5, enable HiDPI if provided by qdarktheme (safe no-op otherwise)
    try:
        if qdarktheme is not None and hasattr(qdarktheme, "enable_hi_dpi"):
            qdarktheme.enable_hi_dpi()
    except Exception:
        pass

    app = QApplication(sys.argv)

    # Apply initial theme via PyQtDarkTheme (Light by default)
    try:
        if qdarktheme is not None:
            qdarktheme.setup_theme("light")
    except Exception:
        pass

    # Icon for app
    try:
        from PyQt5.QtGui import QIcon
        from pathlib import Path
        icon_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    window = PanoramaAppWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


