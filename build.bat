@echo off
echo ===============================================================
echo  Building Experiment 5: Neural Network Layers in CUDA
echo ===============================================================

where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvcc was not found in PATH!
    echo Please run this script inside the Developer Command Prompt for VS.
    exit /b 1
)

where cl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cl.exe was not found in PATH!
    echo Please open the Developer Command Prompt for VS 2022/2026.
    exit /b 1
)

echo Compiling src\neural_network.cu ...
nvcc -O2 src\neural_network.cu -o bin\neural_network.exe

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Build succeeded: bin\neural_network.exe
) else (
    echo [FAILED] Compilation failed.
    exit /b %ERRORLEVEL%
)
