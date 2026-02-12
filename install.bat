@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo [INFO] Создание виртуального окружения...
python -m venv venv

echo [INFO] Активация окружения...
call venv\Scripts\activate

echo [INFO] Обновление pip...
python -m pip install --upgrade pip

echo [INFO] Очистка кэша pip...
python -m pip cache purge

echo [INFO] Установка Python-зависимостей...
python -m pip install -r requirements.txt

echo [INFO] Патч basicsr...
python -m pip install basicsr-fixed

echo [INFO] Удаление opencv-python...
python -m pip uninstall opencv-python opencv-python-headless -y
echo [INFO] Копирование кастомного OpenCV (cv2 + sitecustomize)...

xcopy /Y /Q "custom-cv2\cv2.cp310-win_amd64.pyd" "venv\Lib\site-packages\"
if errorlevel 1 (
    echo [ERROR] Не удалось скопировать cv2.pyd!
    pause
    exit /b 1
)

xcopy /Y /Q "custom-cv2\sitecustomize.py" "venv\Lib\site-packages\"
if errorlevel 1 (
    echo [WARNING] Не удалось скопировать sitecustomize.py (не критично)
)

echo [SUCCESS] Кастомный OpenCV успешно установлен в venv.
echo [NOTE] Папку 'custom-cv2' теперь можно безопасно удалить.

echo [INFO] Проверка загрузки cv2 и CUDA...
python -c "import cv2; print('✅ OpenCV version:', cv2.__version__); print('CUDA:', 'Yes' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'No')"

echo [INFO] Проверка PyTorch CUDA...
python -c "import torch; print('✅ Torch CUDA available:', torch.cuda.is_available())"

echo.
echo [INSTALL COMPLETE] Запуск приложения: run.bat
pause