@echo off
setlocal
echo ==========================================================
echo Push SMS Spam Detection Project to GitHub
echo ==========================================================
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g., https://github.com/username/sms-spam-detection.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL provided. Aborting.
    pause
    exit /b 1
)

echo.
echo Setting remote origin to %REPO_URL%...
"%LOCALAPPDATA%\Programs\Git\cmd\git.exe" remote remove origin 2>nul
"%LOCALAPPDATA%\Programs\Git\cmd\git.exe" remote add origin %REPO_URL%

echo.
echo Pushing branch 'main' to GitHub...
"%LOCALAPPDATA%\Programs\Git\cmd\git.exe" push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================================
    echo SUCCESS! Repository pushed to GitHub successfully.
    echo You can now send this link to your professor:
    echo %REPO_URL%
    echo ==========================================================
) else (
    echo.
    echo [ERROR] Push failed. Please check your credentials or repo permissions.
)

pause
