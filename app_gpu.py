# === Настройка CUDA_DLL перед всеми импортами ===
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_DLL_DIR = os.path.join(SCRIPT_DIR, "cuda-dll")
if os.path.isdir(CUDA_DLL_DIR):
    os.environ["PATH"] = CUDA_DLL_DIR + os.pathsep + os.environ.get("PATH", "")
    os.environ["CUDA_PATH"] = CUDA_DLL_DIR
    print(f"[INFO] CUDA_PATH установлен: {CUDA_DLL_DIR}")
else:
    print(f"[WARNING] Папка cuda-dll не найдена!")
# === Основные импорты ===
import warnings
warnings.filterwarnings("ignore", module="albumentations.*")
warnings.filterwarnings("ignore", module="torchvision.*")
import cv2
import numpy as np
import threading
import time
import queue
import torch
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from settings import (
    CAPTURE_WIDTH, CAPTURE_HEIGHT, MAX_QUEUE_SIZE,
    DEFAULT_PROVIDER, DEFAULT_SHARPNESS, DEFAULT_MASK_BLUR,
    DEFAULT_OVAL_WIDTH, DEFAULT_OVAL_HEIGHT,
    PROVIDERS, INSWAPPER_PATH, INSIGHTFACE_ROOT, GFPGAN_MODEL_PATH
)
from ui import MainWindow
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from gfpgan import GFPGANer

class FaceSwapperApp(QObject):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.show_fps = False
        self.use_gfpgan = False
        self.last_selected_photo_path = None
        self.source_face = None
        self.face_analyser = None
        self.face_swapper = None
        self.gfpgan_model = None
        self.current_provider = DEFAULT_PROVIDER
        self.sharpness_value = DEFAULT_SHARPNESS
        self.mask_blur_amount = DEFAULT_MASK_BLUR
        self.oval_width = DEFAULT_OVAL_WIDTH
        self.oval_height = DEFAULT_OVAL_HEIGHT
        
        # Разрешение камеры (динамическое)
        self.capture_width = CAPTURE_WIDTH
        self.capture_height = CAPTURE_HEIGHT
        
        # Индекс текущей камеры
        self.camera_index = 0
        
        # ── Оптимизации ───────────────────────────────────────
        self.last_faces = None
        self.frame_counter = 0
        self.DETECT_EVERY_N = 2  # Увеличено с 1 до 2 - детекция через кадр
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        
        # ── Кэширование масок по размеру лица ─────────────────
        self.mask_cache = {}
        
        # ── Пул для блюра масок (предрасчет) ──────────────────
        self.blurred_mask_cache = {}
        
        # Профилирование
        self.prof_detect = 0.0
        self.prof_swap = 0.0
        self.prof_post = 0.0
        self.prof_total = 0.0
        self.prof_count = 0
        
        # Потоковая безопасность для изменения разрешения
        self.resolution_lock = threading.Lock()
        self.restart_capture = False
        
        if not self.init_models():
            sys.exit(1)
        
        # Обнаружение доступных камер
        available_cameras = self.detect_cameras()
        
        self.window = MainWindow()
        self._connect_signals()
        
        # Заполняем список камер
        self.window.populate_cameras(available_cameras)
        self.window.set_camera_by_index(self.camera_index)
        
        # === ИСПРАВЛЕНИЕ 1: Правильная установка начальных значений слайдеров ===
        self.window.set_slider_value("sharpness", DEFAULT_SHARPNESS, 0.1)
        self.window.set_slider_value("blur", DEFAULT_MASK_BLUR, 2)
        self.window.set_slider_value("width", DEFAULT_OVAL_WIDTH, 0.05)
        self.window.set_slider_value("height", DEFAULT_OVAL_HEIGHT, 0.05)
        
        # Устанавливаем разрешение из settings в UI
        self.window.set_resolution_by_size(CAPTURE_WIDTH, CAPTURE_HEIGHT)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)
    
    def detect_cameras(self, max_cameras=10):
        """Обнаруживает доступные веб-камеры
        
        Args:
            max_cameras: максимальное количество камер для проверки
            
        Returns:
            list: список кортежей (index, name) доступных камер
        """
        available_cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Пытаемся получить информацию о камере
                ret, _ = cap.read()
                if ret:
                    # Получаем разрешение камеры для отображения
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    camera_name = f"Камера {i} ({width}x{height})"
                    available_cameras.append((i, camera_name))
                cap.release()
        
        # Если не найдено ни одной камеры, добавляем камеру по умолчанию
        if not available_cameras:
            available_cameras.append((0, "Камера 0 (по умолчанию)"))
            print("[WARNING] Камеры не обнаружены, используется камера 0 по умолчанию")
        else:
            print(f"[INFO] Обнаружено камер: {len(available_cameras)}")
            for idx, name in available_cameras:
                print(f"  - {name}")
        
        return available_cameras

    def _connect_signals(self):
        w = self.window
        w.select_photo_requested.connect(self.select_or_update_photo)
        w.toggle_gfpgan_requested.connect(self.toggle_gfpgan)
        w.toggle_fps_requested.connect(self.toggle_fps)
        w.start_stop_requested.connect(self.toggle_start)
        w.sharpness_changed.connect(self.update_sharpness)
        w.mask_blur_changed.connect(self.update_mask_blur)
        w.oval_width_changed.connect(self.update_oval_width)
        w.oval_height_changed.connect(self.update_oval_height)
        w.resolution_changed.connect(self.update_resolution)
        w.camera_changed.connect(self.update_camera)

    def init_models(self):
        try:
            providers = PROVIDERS[self.current_provider]
            print(f"Loading models with provider: {self.current_provider}")
            
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                root=INSIGHTFACE_ROOT,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            # ОПТИМИЗАЦИЯ: Уменьшен размер детекции для ускорения
            self.face_analyser.prepare(ctx_id=0, det_size=(128, 128))  # Было 160x160
            
            print(f"Loading swapper: {INSWAPPER_PATH}")
            self.face_swapper = model_zoo.get_model(INSWAPPER_PATH, providers=providers)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gfpgan_model = GFPGANer(
                model_path=GFPGAN_MODEL_PATH,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=device
            )
            
            print("✅ Models loaded successfully.")
            return True
        except Exception as e:
            if hasattr(self, 'window'):
                self.window.show_error("Ошибка", f"Не удалось загрузить модели:\n{str(e)}")
            else:
                print(f"Model load error: {e}")
            return False

    def get_mask_template(self, face_w, face_h):
        """Кэширует шаблон маски по размеру лица"""
        key = (face_w, face_h, self.oval_width, self.oval_height)
        if key not in self.mask_cache:
            mask = np.zeros((face_h, face_w), dtype=np.uint8)
            cx, cy = face_w // 2, face_h // 2
            rx = int(face_w * self.oval_width)
            ry = int(face_h * self.oval_height)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            self.mask_cache[key] = mask
        return self.mask_cache[key]
    
    def get_blurred_mask(self, face_w, face_h, blur_amount):
        """ОПТИМИЗАЦИЯ: Кэширование размытых масок"""
        # Гарантируем нечетное значение для GaussianBlur
        blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        key = (face_w, face_h, self.oval_width, self.oval_height, blur_amount)
        if key not in self.blurred_mask_cache:
            # Получаем базовую маску
            mask = self.get_mask_template(face_w, face_h)
            # Размываем её более интенсивно для плавного перехода
            blurred = cv2.GaussianBlur(mask, (blur_amount, blur_amount), blur_amount / 3.0)
            self.blurred_mask_cache[key] = blurred
            # Ограничиваем размер кэша
            if len(self.blurred_mask_cache) > 20:
                self.blurred_mask_cache.pop(next(iter(self.blurred_mask_cache)))
        return self.blurred_mask_cache[key]

    def apply_sharpness_to_region(self, image, x1, y1, x2, y2, strength):
        """ОПТИМИЗАЦИЯ: Улучшен алгоритм резкости"""
        if strength <= 0:
            return image
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        
        region = image[y1:y2, x1:x2]
        try:
            # ОПТИМИЗАЦИЯ: Используем более быстрый метод
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(region)
            gpu_blur = cv2.cuda.GaussianBlur(gpu_img, (0, 0), 2)  # Уменьшена сигма с 3 до 2
            gpu_sharp = cv2.cuda.addWeighted(gpu_img, 1 + strength, gpu_blur, -strength, 0)
            sharpened = gpu_sharp.download()
        except:
            # CPU fallback
            blurred = cv2.GaussianBlur(region, (0, 0), 2)
            sharpened = cv2.addWeighted(region, 1 + strength, blurred, -strength, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        image[y1:y2, x1:x2] = sharpened
        return image

    def process_photo_with_gfpgan(self, img):
        if self.use_gfpgan and self.gfpgan_model is not None:
            try:
                _, _, enhanced = self.gfpgan_model.enhance(
                    img.copy(),
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True
                )
                return enhanced
            except Exception:
                pass
        return img

    def select_or_update_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Выберите фото с лицом",
            "", "Image files (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self.last_selected_photo_path = path
        img = cv2.imread(path)
        if img is None:
            self.window.show_error("Ошибка", "Не удалось загрузить изображение.")
            return
        processed_img = self.process_photo_with_gfpgan(img)
        faces = self.face_analyser.get(processed_img)
        if not faces:
            self.window.show_error("Внимание", "Лицо на фото не найдено!")
            return
        self.source_face = faces[0]
        self.window.update_preview(processed_img)

    def toggle_gfpgan(self):
        self.use_gfpgan = not self.use_gfpgan
        self.window.set_gfpgan_state(self.use_gfpgan)
        if self.last_selected_photo_path:
            img = cv2.imread(self.last_selected_photo_path)
            if img is not None:
                processed = self.process_photo_with_gfpgan(img)
                faces = self.face_analyser.get(processed)
                if faces:
                    self.source_face = faces[0]
                    self.window.update_preview(processed)

    def toggle_fps(self):
        self.show_fps = not self.show_fps

    def update_sharpness(self, val: float):
        self.sharpness_value = val

    def update_mask_blur(self, val: int):
        self.mask_blur_amount = val if val % 2 == 1 else val + 1
        # Очистка кэша размытых масок при изменении
        self.blurred_mask_cache.clear()

    def update_oval_width(self, val: float):
        self.oval_width = val
        self.mask_cache.clear()
        self.blurred_mask_cache.clear()

    def update_oval_height(self, val: float):
        self.oval_height = val
        self.mask_cache.clear()
        self.blurred_mask_cache.clear()
    
    def update_resolution(self, width: int, height: int):
        """Обновляет разрешение камеры"""
        with self.resolution_lock:
            self.capture_width = width
            self.capture_height = height
            # Если камера работает, перезапускаем захват
            if self.running:
                self.restart_capture = True
                print(f"[INFO] Разрешение изменено на {width}x{height}, перезапуск захвата...")
    
    def update_camera(self, camera_index: int):
        """Обновляет индекс используемой камеры"""
        with self.resolution_lock:
            self.camera_index = camera_index
            # Если камера работает, перезапускаем захват
            if self.running:
                self.restart_capture = True
                print(f"[INFO] Переключение на камеру {camera_index}, перезапуск захвата...")

    def toggle_start(self):
        if not self.running:
            if self.source_face is None:
                self.window.show_error("Внимание", "Сначала выберите фото с лицом!")
                return
            self.running = True
            self.window.set_start_button(True)
            self.window.set_resolution_combo_enabled(False)  # Блокируем изменение разрешения
            self.window.set_camera_combo_enabled(False)  # Блокируем изменение камеры
            threading.Thread(target=self.capture_thread, daemon=True).start()
            threading.Thread(target=self.process_thread, daemon=True).start()
        else:
            self.running = False
            self.window.set_start_button(False)
            self.window.set_resolution_combo_enabled(True)  # Разблокируем
            self.window.set_camera_combo_enabled(True)  # Разблокируем

    def capture_thread(self):
        """ОПТИМИЗАЦИЯ: Улучшенный поток захвата с поддержкой динамического разрешения и выбора камеры"""
        cap = None
        
        while self.running:
            # Открываем/переоткрываем камеру при необходимости
            if cap is None or self.restart_capture:
                if cap is not None:
                    cap.release()
                
                # Используем динамический индекс камеры
                with self.resolution_lock:
                    current_camera = self.camera_index
                    current_width = self.capture_width
                    current_height = self.capture_height
                
                cap = cv2.VideoCapture(current_camera)
                if not cap.isOpened():
                    self.window.show_error("Ошибка", f"Не удалось открыть камеру {current_camera}.")
                    self.running = False
                    return
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
                # ОПТИМИЗАЦИЯ: Пытаемся установить FPS камеры
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Проверяем реальное разрешение
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[INFO] Камера {current_camera}: запрошено {current_width}x{current_height}, "
                      f"получено {actual_w}x{actual_h}")
                self.restart_capture = False
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # ОПТИМИЗАЦИЯ: Пропускаем кадры если очередь полная
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # Минимальная задержка для снижения нагрузки
            time.sleep(0.001)
        
        if cap is not None:
            cap.release()

    def process_thread(self):
        """ОПТИМИЗАЦИЯ: Улучшенный поток обработки"""
        prev_time = time.time()
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            
            # ── Пропускаем детекцию каждые N кадров ──
            self.frame_counter += 1
            if self.frame_counter % self.DETECT_EVERY_N == 0 or self.last_faces is None:
                self.last_faces = self.face_analyser.get(frame)
            faces = self.last_faces if self.last_faces is not None else []
            t1 = time.perf_counter()

            h, w = frame.shape[:2]
            output_frame = frame  # ОПТИМИЗАЦИЯ: Работаем напрямую с frame

            t_swap_start = time.perf_counter()
            
            # ── Обработка лиц ──
            for face in faces:
                try:
                    bbox = face.bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_w, face_h = x2 - x1, y2 - y1

                    # ── Сваппинг ──
                    temp_swapped = self.face_swapper.get(
                        output_frame,
                        face,
                        self.source_face,
                        paste_back=True
                    )
                    
                    t_swap_end = time.perf_counter()

                    # ── Создаём и размываем маску ──
                    if self.mask_blur_amount > 1:
                        mask_local = self.get_blurred_mask(face_w, face_h, self.mask_blur_amount)
                    else:
                        mask_local = self.get_mask_template(face_w, face_h)
                    
                    # ── Применяем шарпнинг ТОЛЬКО к области лица ──
                    if self.sharpness_value > 0:
                        temp_swapped = self.apply_sharpness_to_region(
                            temp_swapped, x1, y1, x2, y2, self.sharpness_value
                        )

                    # ── Блендинг только области лица с feathering ──
                    # ОПТИМИЗАЦИЯ: Используем vectorized операции с улучшенным feathering
                    mask_f = mask_local.astype(np.float32) / 255.0
                    # Применяем power для более плавного перехода (gamma correction)
                    mask_f = np.power(mask_f, 0.8)
                    mask_3d = np.stack([mask_f, mask_f, mask_f], axis=2)
                    
                    orig_region = output_frame[y1:y2, x1:x2].astype(np.float32)
                    swapped_region = temp_swapped[y1:y2, x1:x2].astype(np.float32)
                    
                    # ОПТИМИЗАЦИЯ: Одна операция вместо двух
                    blended = (swapped_region * mask_3d + orig_region * (1.0 - mask_3d)).astype(np.uint8)
                    output_frame[y1:y2, x1:x2] = blended

                except Exception as e:
                    print(f"Error in face processing: {e}")
                    continue

            t2 = time.perf_counter()

            # ── Профилирование с выводом реального FPS ──
            total_time = t2 - t0
            current_fps = 1.0 / total_time if total_time > 0 else 0.0
            
            self.prof_detect += (t1 - t0)
            self.prof_swap   += (t_swap_end - t_swap_start) if len(faces) > 0 else 0.0
            self.prof_post   += (t2 - t1)
            self.prof_total  += total_time
            self.prof_count  += 1
            
            if self.prof_count >= 60:
                avg_detect = self.prof_detect / self.prof_count * 1000
                avg_swap   = self.prof_swap   / self.prof_count * 1000
                avg_post   = self.prof_post   / self.prof_count * 1000
                avg_total  = self.prof_total  / self.prof_count * 1000
                avg_fps    = self.prof_count / self.prof_total if self.prof_total > 0 else 0.0
                
                print(f"[PROF] detect: {avg_detect:5.1f} ms | "
                      f"swap: {avg_swap:5.1f} ms | "
                      f"post: {avg_post:5.1f} ms | "
                      f"total: {avg_total:5.1f} ms | fps: {avg_fps:.1f}")
                
                self.prof_detect = self.prof_swap = self.prof_post = self.prof_total = 0.0
                self.prof_count = 0

            # ── Отображение FPS на кадре ──
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            if self.show_fps:
                cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ОПТИМИЗАЦИЯ: Пропускаем кадр если очередь результатов полная
            if not self.result_queue.full():
                self.result_queue.put(output_frame)

    def update_gui(self):
        if not self.running:
            return
        try:
            frame = self.result_queue.get_nowait()
            self.frame_ready.emit(frame)
        except queue.Empty:
            pass

    def run(self):
        self.window.show()
        self.frame_ready.connect(self.window.update_video)
        return QApplication.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    swapper = FaceSwapperApp()
    sys.exit(swapper.run())