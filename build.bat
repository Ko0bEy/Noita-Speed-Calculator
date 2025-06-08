@echo off
REM Force working directory to location of this script
cd /d %~dp0

REM venv is assumed to be in the same directory as this script
set "VENV=%CD%\venv"

REM Set target directory (customize)
set "TARGETDIR=D:\speed_calc"

REM Remove old .spec files if present (to avoid permission issues)
del /F /Q "angle_solver.spec" 2>nul
del /F /Q "speed_calc.spec" 2>nul

REM Use the venv's python to run pyinstaller
"%VENV%\Scripts\python.exe" -m PyInstaller --onefile --hidden-import=matplotlib.backends --hidden-import=matplotlib.backends.backend_tkagg angle_solver.py
"%VENV%\Scripts\python.exe" -m PyInstaller --onefile --hidden-import=matplotlib.backends --hidden-import=matplotlib.backends.backend_tkagg speed_calc.py

REM Create target directory if it doesn't exist
if not exist "%TARGETDIR%" mkdir "%TARGETDIR%"

REM Copy the EXEs to the target directory
copy /Y "dist\angle_solver.exe" "%TARGETDIR%" >nul
copy /Y "dist\speed_calc.exe" "%TARGETDIR%" >nul

echo Build and copy complete.
pause
