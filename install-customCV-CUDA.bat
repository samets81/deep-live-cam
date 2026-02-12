@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Deep Live Cam TensorRT - Установка
echo ========================================
echo.

REM Проверка существования папок
set NEED_DOWNLOAD=0

if not exist "custom-cv2" (
    echo [INFO] Папка custom-cv2 не найдена, требуется скачивание.
    set NEED_DOWNLOAD=1
) else (
    echo [OK] Папка custom-cv2 уже существует.
)

if not exist "cuda-dll" (
    echo [INFO] Папка cuda-dll не найдена, требуется скачивание.
    set NEED_DOWNLOAD=1
) else (
    echo [OK] Папка cuda-dll уже существует.
)

if !NEED_DOWNLOAD! equ 0 (
    echo [INFO] Все необходимые папки уже существуют, пропускаем скачивание.
    echo.
    goto SKIP_DOWNLOAD
)

REM Проверка наличия curl
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] curl не найден! Установите curl или Windows 10+ для продолжения.
    pause
    exit /b 1
)

if not exist "custom-cv2" (
    echo [INFO] Скачивание custom-cv2.zip...
    curl -L -o custom-cv2.zip https://minyakov.ru/custom-cv2.zip
    if errorlevel 1 (
        echo [ERROR] Не удалось скачать custom-cv2.zip!
        pause
        exit /b 1
    )
    
    echo [INFO] Распаковка custom-cv2.zip...
    powershell -command "Expand-Archive -Path 'custom-cv2.zip' -DestinationPath 'custom-cv2' -Force"
    if errorlevel 1 (
        echo [ERROR] Не удалось распаковать custom-cv2.zip!
        del custom-cv2.zip 2>nul
        pause
        exit /b 1
    )
    
    echo [INFO] Удаление архива custom-cv2.zip...
    del custom-cv2.zip
)

if not exist "cuda-dll" (
    echo [INFO] Скачивание cuda-dll.zip...
    curl -L -o cuda-dll.zip https://minyakov.ru/cuda-dll.zip
    if errorlevel 1 (
        echo [ERROR] Не удалось скачать cuda-dll.zip!
        pause
        exit /b 1
    )
    
    echo [INFO] Распаковка cuda-dll.zip...
    powershell -command "Expand-Archive -Path 'cuda-dll.zip' -DestinationPath 'cuda-dll' -Force"
    if errorlevel 1 (
        echo [ERROR] Не удалось распаковать cuda-dll.zip!
        del cuda-dll.zip 2>nul
        rd /s /q custom-cv2 2>nul
        pause
        exit /b 1
    )
    
    echo [INFO] Удаление архива cuda-dll.zip...
    del cuda-dll.zip
)

echo [SUCCESS] Архивы успешно скачаны и распакованы!
echo.

:SKIP_DOWNLOAD

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
echo ========================================
echo [INSTALL COMPLETE] 
echo Запуск приложения: run.bat
echo ========================================
pause

