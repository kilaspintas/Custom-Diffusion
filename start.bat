@echo off
set HF_HOME=%~dp0model
set GRADIO_ANALYTICS_ENABLED=False
set PYTHONUNBUFFERED=1

conda env list | findstr /B "IPEX " >nul
IF %ERRORLEVEL% EQU 0 (
    goto :RUN_APP
) ELSE (
    goto :FIRST_SETUP
)

:FIRST_SETUP
echo.
echo ========================================================
echo  WELCOME! RUNNING FIRST-TIME SETUP FOR INTEL IPEX...
echo ========================================================
echo.
echo This process will create a Conda environment and install
echo all required dependencies. This will take some time and
echo requires an internet connection.
echo Please do not close this window.
echo.
pause
echo.

echo [1/3] Creating 'IPEX' environment and installing standard dependencies...
call conda env create -f environment.yml
IF %ERRORLEVEL% NEQ 0 ( 
    echo FAILED: Could not create environment from environment.yml.
    pause 
    exit /b 
)

echo. & echo --- Activating environment for special installation ---
call conda activate IPEX
IF %ERRORLEVEL% NEQ 0 ( echo FAILED: Conda environment activation failed. & pause & exit /b )

echo. & echo [2/3] Installing PyTorch and IPEX...
echo This command might uninstall a CPU-only torch version if it exists and replace it.
call python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
IF %ERRORLEVEL% NEQ 0 ( echo FAILED: PyTorch installation failed. & pause & exit /b )
call python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
IF %ERRORLEVEL% NEQ 0 ( echo FAILED: IPEX installation failed. & pause & exit /b )

echo. & echo [3/3] Verifying installation...
call python -c "import intel_extension_for_pytorch as ipex; import gradio; print(f'IPEX version: {ipex.__version__}'); print(f'Gradio version: {gradio.__version__}')"
IF %ERRORLEVEL% NEQ 0 ( echo FAILED: Dependency verification failed. & pause & exit /b )

echo. & echo ===================================
echo   SETUP FOR INTEL COMPLETED!
echo ===================================
pause
goto :RUN_APP

:RUN_APP
echo. & echo --- Starting application with Intel (IPEX) backend ---
call conda activate IPEX
echo.
echo Starting application...
echo.
call python app.py
goto :END_SCRIPT

:END_SCRIPT
echo.
echo Application has been closed.
pause