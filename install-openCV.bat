@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Deep Live Cam TensorRT - Установка
echo ========================================
echo.

REM Проверка существования папок
set NEED_DOWNLOAD=0


if not exist "cuda-dll" (
    echo [INFO] Папка cuda-dll не найдена, требуется скачивание.
    set NEED_DOWNLOAD=1
) else (
    echo [OK] Папка cuda-dll уже существует.
)

if !NEED_DOWNLOAD! equ 0 (
    echo [INFO] Пропускаем скачивание.
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

echo [SUCCESS] Архив успешно скачан и распакован!
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


echo.
echo ========================================
echo [INSTALL COMPLETE] 
echo Запуск приложения: run.bat
echo ========================================
pause
