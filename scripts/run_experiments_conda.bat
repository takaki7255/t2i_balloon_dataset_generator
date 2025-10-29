@echo off
REM Anaconda環境で実験を実行するバッチファイル
REM 使用方法: run_experiments_conda.bat myenv [all|syn200-corner,syn500-corner]

setlocal enabledelayedexpansion

REM ============================================================================
REM 引数チェック
REM ============================================================================

if "%~1"=="" (
    echo ❌ 使用方法: %~nx0 ^<conda_env^> [datasets]
    echo.
    echo 例:
    echo   %~nx0 myenv
    echo   %~nx0 myenv syn200-corner,syn500-corner
    echo   %~nx0 myenv syn500-corner
    echo.
    exit /b 1
)

set CONDA_ENV=%~1
set DATASETS=%~2
if "%DATASETS%"=="" set DATASETS=all

echo ================================================================================
echo 🐍 Conda環境: %CONDA_ENV%
echo 📊 データセット: %DATASETS%
echo ================================================================================
echo.

REM ============================================================================
REM Conda環境の確認
REM ============================================================================

echo 🔍 Conda環境を確認中...
conda env list | findstr /B "%CONDA_ENV% " >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda環境 '%CONDA_ENV%' が見つかりません。
    echo.
    echo 利用可能な環境:
    conda env list
    exit /b 1
)
echo ✅ Conda環境 '%CONDA_ENV%' が見つかりました
echo.

REM ============================================================================
REM PowerShellスクリプトを実行
REM ============================================================================

echo 🚀 実験を開始します...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0run_experiments_conda.ps1" -CondaEnv "%CONDA_ENV%" -Datasets "%DATASETS%"

if errorlevel 1 (
    echo.
    echo ❌ 実験が失敗しました
    exit /b 1
)

echo.
echo ✅ 実験が完了しました
exit /b 0
