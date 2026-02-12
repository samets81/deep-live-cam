from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QGroupBox, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


class VideoDisplayLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setText("Видео с камеры")
        self.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")

    def set_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


class VideoWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preview - Face Swap")
        self.resize(660, 500)
        self.video_label = VideoDisplayLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)
        self.setModal(False)

    def set_frame(self, frame: np.ndarray):
        self.video_label.set_frame(frame)
    
    def update_size(self, width: int, height: int):
        """Обновляет размер окна превью под разрешение камеры"""
        # Добавляем небольшой отступ
        self.resize(width + 20, height + 40)


class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Превью\nисходного лица")
        self.setStyleSheet("border: 1px solid #aaa; background: #eee;")


class MainWindow(QMainWindow):
    select_photo_requested = pyqtSignal()
    toggle_gfpgan_requested = pyqtSignal()
    toggle_fps_requested = pyqtSignal()
    start_stop_requested = pyqtSignal()
    sharpness_changed = pyqtSignal(float)
    mask_blur_changed = pyqtSignal(int)
    oval_width_changed = pyqtSignal(float)
    oval_height_changed = pyqtSignal(float)
    resolution_changed = pyqtSignal(int, int)  # width, height
    camera_changed = pyqtSignal(int)  # camera_index

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Swapper + GFPGAN (TensorRT)")
        self.setFixedSize(480, 820)  # Увеличили высоту для нового элемента
        self.video_window = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Выбрать фото")
        self.btn_gfpgan = QPushButton("GFPGAN: ВЫКЛ")
        self.btn_fps = QPushButton("Показать FPS")
        self.btn_start = QPushButton("Старт")

        self.btn_select.clicked.connect(self.select_photo_requested)
        self.btn_gfpgan.clicked.connect(self.toggle_gfpgan_requested)
        self.btn_fps.clicked.connect(self._on_fps_clicked)
        self.btn_start.clicked.connect(self.start_stop_requested)

        for btn in [self.btn_select, self.btn_gfpgan, self.btn_fps, self.btn_start]:
            btn.setFixedHeight(32)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_gfpgan)
        btn_layout.addWidget(self.btn_fps)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        # Camera selector
        self._add_camera_selector(layout)

        # Resolution selector
        self._add_resolution_selector(layout)

        # Sliders
        self._add_slider_with_value(layout, "Резкость лица", 0.0, 5.0, 0.1, self.sharpness_changed, "sharpness")
        self._add_slider_with_value(layout, "Плавность перехода", 1, 99, 2, self.mask_blur_changed, "blur")
        self._add_slider_with_value(layout, "Ширина овала", 0.4, 1.5, 0.05, self.oval_width_changed, "width")
        self._add_slider_with_value(layout, "Высота овала", 0.4, 1.5, 0.05, self.oval_height_changed, "height")

        # Status & Preview
        self.status_label = QLabel("GFPGAN: выключен")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.status_label)
        self.preview_label = PreviewLabel()
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        self.setCentralWidget(central)
        self.video_window = VideoWindow()

    def _add_camera_selector(self, parent_layout):
        """Добавляет выбор веб-камеры"""
        group = QGroupBox("Веб-камера")
        layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        
        layout.addWidget(QLabel("Выберите:"))
        layout.addWidget(self.camera_combo)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _on_camera_changed(self, index):
        """Обработчик изменения камеры"""
        camera_index = self.camera_combo.itemData(index)
        if camera_index is not None:
            self.camera_changed.emit(camera_index)

    def _add_resolution_selector(self, parent_layout):
        """Добавляет выбор разрешения камеры"""
        group = QGroupBox("Разрешение камеры")
        layout = QHBoxLayout()
        
        self.resolution_combo = QComboBox()
        # Популярные разрешения веб-камер
        resolutions = [
            ("320x240", 320, 240),
            ("640x480 (VGA)", 640, 480),
            ("800x600", 800, 600),
            ("1024x768", 1024, 768),
            ("1280x720 (HD)", 1280, 720),
            ("1920x1080 (FHD)", 1920, 1080),
        ]
        
        for text, width, height in resolutions:
            self.resolution_combo.addItem(text, (width, height))
        
        # По умолчанию 640x480
        self.resolution_combo.setCurrentIndex(1)
        
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        
        layout.addWidget(QLabel("Выберите:"))
        layout.addWidget(self.resolution_combo)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _on_resolution_changed(self, index):
        """Обработчик изменения разрешения"""
        width, height = self.resolution_combo.itemData(index)
        self.resolution_changed.emit(width, height)
        # Обновляем размер окна превью
        if self.video_window:
            self.video_window.update_size(width, height)

    def _add_slider_with_value(self, parent_layout, label_text, min_val, max_val, step, callback_signal, name):
        group = QGroupBox(label_text)
        slider = QSlider(Qt.Horizontal)
        int_min = int(min_val / step)
        int_max = int(max_val / step)
        slider.setRange(int_min, int_max)
        slider.setSingleStep(1)
        value_label = QLabel(str(min_val))
        value_label.setAlignment(Qt.AlignCenter)

        def on_slider_change(v):
            real_val = v * step
            formatted = f"{real_val:.2f}" if isinstance(step, float) and step < 1 else str(int(real_val))
            value_label.setText(formatted)
            callback_signal.emit(real_val)

        slider.valueChanged.connect(on_slider_change)
        vbox = QVBoxLayout()
        vbox.addWidget(slider)
        vbox.addWidget(value_label)
        group.setLayout(vbox)
        parent_layout.addWidget(group)
        setattr(self, f"slider_{name}", slider)
        setattr(self, f"label_{name}_value", value_label)
    
    def set_slider_value(self, name: str, value: float, step: float):
        """Устанавливает значение слайдера и обновляет label"""
        slider = getattr(self, f"slider_{name}")
        value_label = getattr(self, f"label_{name}_value")
        int_value = int(value / step)
        # Блокируем сигналы чтобы не было двойного вызова
        slider.blockSignals(True)
        slider.setValue(int_value)
        slider.blockSignals(False)
        # Обновляем label вручную
        formatted = f"{value:.2f}" if isinstance(step, float) and step < 1 else str(int(value))
        value_label.setText(formatted)

    def _on_fps_clicked(self):
        """Обработчик нажатия кнопки FPS - меняет текст и отправляет сигнал"""
        if self.btn_fps.text() == "Показать FPS":
            self.btn_fps.setText("Скрыть FPS")
        else:
            self.btn_fps.setText("Показать FPS")
        self.toggle_fps_requested.emit()

    def set_gfpgan_state(self, enabled: bool):
        text = "GFPGAN: ВКЛ" if enabled else "GFPGAN: ВЫКЛ"
        color = "lightgreen" if enabled else "lightcoral"
        self.btn_gfpgan.setText(text)
        self.btn_gfpgan.setStyleSheet(f"background-color: {color};")
        status_text = "GFPGAN: включён" if enabled else "GFPGAN: выключен"
        status_color = "green" if enabled else "red"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

    def set_start_button(self, running: bool):
        text = "Стоп" if running else "Старт"
        color = "lightcoral" if running else "lightgreen"
        self.btn_start.setText(text)
        self.btn_start.setStyleSheet(f"background-color: {color};")
        if running:
            self.video_window.show()
        else:
            self.video_window.hide()
    
    def set_resolution_combo_enabled(self, enabled: bool):
        """Блокирует/разблокирует выбор разрешения во время работы"""
        self.resolution_combo.setEnabled(enabled)
    
    def set_camera_combo_enabled(self, enabled: bool):
        """Блокирует/разблокирует выбор камеры во время работы"""
        self.camera_combo.setEnabled(enabled)
    
    def populate_cameras(self, cameras: list):
        """Заполняет список доступных камер
        
        Args:
            cameras: список кортежей (index, name)
        """
        self.camera_combo.clear()
        for index, name in cameras:
            self.camera_combo.addItem(name, index)
    
    def set_camera_by_index(self, camera_index: int):
        """Устанавливает камеру в комбобоксе по индексу"""
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == camera_index:
                self.camera_combo.setCurrentIndex(i)
                break

    def get_current_resolution(self):
        """Возвращает текущее выбранное разрешение"""
        return self.resolution_combo.currentData()
    
    def set_resolution_by_size(self, width: int, height: int):
        """Устанавливает разрешение в комбобоксе по размеру"""
        for i in range(self.resolution_combo.count()):
            w, h = self.resolution_combo.itemData(i)
            if w == width and h == height:
                self.resolution_combo.setCurrentIndex(i)
                break

    def update_preview(self, image: np.ndarray):
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(200, 200, Qt.KeepAspectRatio)
        self.preview_label.setPixmap(pixmap)

    def update_video(self, frame: np.ndarray):
        if self.video_window.isVisible():
            self.video_window.set_frame(frame)

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str):
        QMessageBox.information(self, title, message)