@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_vls.ps1" %*
set "VLS_EXIT_CODE=%ERRORLEVEL%"

if not "%VLS_EXIT_CODE%"=="0" (
    echo.
    echo VLS startup failed with exit code %VLS_EXIT_CODE%.
    pause
)

exit /b %VLS_EXIT_CODE%
