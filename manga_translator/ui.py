import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTabWidget, QFormLayout, 
    QLineEdit, QComboBox, QSpinBox, QCheckBox, QMessageBox, QScrollArea,
    QProgressBar, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

from .pipeline import MangaTranslationPipeline
from .models import AppSettings, ImageJobResult
from .model_manager import translation_preset_choices, get_translation_preset

class WorkerThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, action, *args, **kwargs):
        super().__init__()
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.action(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PanelParse")
        self.resize(1200, 800)
        
        self.pipeline = MangaTranslationPipeline()
        self.settings = AppSettings.with_env_defaults()
        self.current_image_path: Optional[str] = None
        self.current_result: Optional[ImageJobResult] = None

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel (Controls & Settings)
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tabs for Main controls and Settings
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        self._build_main_tab()
        self._build_settings_tab()

        # Status & Progress
        self.status_label = QLabel("Idle")
        left_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_panel)

        # Right Panel (Image Display)
        self.image_view = QLabel("No image loaded")
        self.image_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_view.setStyleSheet("background-color: #222; color: #aaa;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_view)
        
        main_layout.addWidget(scroll_area, 1)

    def _build_main_tab(self):
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.clicked.connect(self.analyze_image)
        self.btn_analyze.setEnabled(False)
        layout.addWidget(self.btn_analyze)

        self.btn_translate = QPushButton("Translate & Compose")
        self.btn_translate.clicked.connect(self.translate_image)
        self.btn_translate.setEnabled(False)
        layout.addWidget(self.btn_translate)
        
        layout.addStretch()
        self.tabs.addTab(main_tab, "Actions")

    def _build_settings_tab(self):
        settings_tab = QWidget()
        layout = QFormLayout(settings_tab)
        
        # Translation Preset
        self.cb_trans_preset = QComboBox()
        self.presets = translation_preset_choices()
        self.cb_trans_preset.addItems([label for label, _ in self.presets])
        # Find index for current backend or default
        layout.addRow("Translation Model:", self.cb_trans_preset)

        # Detector Backend
        self.cb_det_backend = QComboBox()
        self.cb_det_backend.addItems(["paddleocr vl 1.5", "paddleocr 3.5", "auto", "default"])
        self.cb_det_backend.setCurrentText(self.settings.detector_backend)
        layout.addRow("Detector Backend:", self.cb_det_backend)

        # Inpainter Backend
        self.cb_inpaint_backend = QComboBox()
        self.cb_inpaint_backend.addItems(["opencv-telea", "opencv-ns", "anime-lama", "migan", "big-lama"])
        self.cb_inpaint_backend.setCurrentText(self.settings.inpainter_backend)
        layout.addRow("Inpainter:", self.cb_inpaint_backend)

        # Font Size
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(8, 96)
        self.spin_font_size.setValue(self.settings.font_size)
        layout.addRow("Font Size:", self.spin_font_size)
        
        def update_translation_preset(idx):
            if idx >= 0 and idx < len(self.presets):
                preset_id = self.presets[idx][1]
                preset = get_translation_preset(preset_id)
                self.settings.translation_backend = preset.backend
                self.settings.translation_model_path = preset.path

        # Update settings on change
        self.cb_trans_preset.currentIndexChanged.connect(update_translation_preset)
        self.cb_det_backend.currentTextChanged.connect(
            lambda t: setattr(self.settings, 'detector_backend', t)
        )
        self.cb_inpaint_backend.currentTextChanged.connect(
            lambda t: setattr(self.settings, 'inpainter_backend', t)
        )
        self.spin_font_size.valueChanged.connect(
            lambda v: setattr(self.settings, 'font_size', v)
        )
        
        self.tabs.addTab(settings_tab, "Settings")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_analyze.setEnabled(True)
            self.btn_translate.setEnabled(False)
            self.status_label.setText(f"Loaded {Path(file_path).name}")

    def display_image(self, path: str):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.image_view.setPixmap(pixmap)
        else:
            self.image_view.setText("Failed to load image")

    def set_busy(self, busy: bool, message: str = ""):
        self.btn_load.setEnabled(not busy)
        self.btn_analyze.setEnabled(not busy and self.current_image_path is not None)
        self.btn_translate.setEnabled(not busy and self.current_result is not None)
        self.tabs.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if message:
            self.status_label.setText(message)

    def analyze_image(self):
        if not self.current_image_path:
            return
            
        self.set_busy(True, "Analyzing...")
        self.worker = WorkerThread(
            self.pipeline.analyze_image, 
            self.current_image_path, 
            self.settings, 
            run_translation=True
        )
        self.worker.finished.connect(self._on_analyze_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_analyze_finished(self, result: ImageJobResult):
        self.current_result = result
        self.set_busy(False, f"Analysis complete. Found {len(result.regions)} regions.")
        self.btn_translate.setEnabled(True)
        if result.overlay_path:
            self.display_image(result.overlay_path)

    def translate_image(self):
        if not self.current_result:
            return
            
        self.set_busy(True, "Translating and composing...")
        self.worker = WorkerThread(
            self.pipeline.compose_image, 
            self.current_image_path, 
            self.settings, 
            self.current_result.regions
        )
        self.worker.finished.connect(self._on_translate_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_translate_finished(self, result: ImageJobResult):
        self.current_result = result
        self.set_busy(False, "Composition complete.")
        if result.final_path:
            self.display_image(result.final_path)

    def _on_worker_error(self, error_msg: str):
        self.set_busy(False, "Error occurred.")
        QMessageBox.critical(self, "Error", error_msg)

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set a dark theme for a modern look
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
