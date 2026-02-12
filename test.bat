@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: Проверка (опционально)
if not exist "cuda-dll\" (
    echo [ERROR] Отсутствует папка cuda-dll
    pause
    exit /b 1
)

:: Настройка PATH
set "PATH=%~dp0cuda-dll"
call venv\Scripts\activate
echo [INFO] Проверка загрузки cv2 и CUDA...
python -c "import cv2; print('✅ OpenCV version:', cv2.__version__); print('CUDA:', 'Yes' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'No')"

echo [INFO] Проверка PyTorch CUDA...
python -c "import torch; print('✅ Torch CUDA available:', torch.cuda.is_available())"

echo.
echo [INFO] Запуск бенчмарка
python benchmark_gpu.py
::python test.py
pause