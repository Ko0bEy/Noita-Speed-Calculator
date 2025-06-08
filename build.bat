@echo off
REM Build both .exe using pyinstaller (adjust .spec/.py if needed)
pyinstaller --onefile angle_solver.py
pyinstaller --onefile speed_calc.py

REM Set target/output directory (adjust if you use something else)
set DISTDIR=dist

REM Name your zip package (e.g., noita_tools_win64.zip)
set ZIPNAME=noita_tools_win64.zip

REM Remove any old zip with same name
if exist "%DISTDIR%\%ZIPNAME%" del "%DISTDIR%\%ZIPNAME%"

REM Go to dist, zip both .exe
cd /d "%DISTDIR%"
REM Use Windows built-in Compress-Archive if on Win 10+ PowerShell, else fallback to 7z if installed

REM Try with PowerShell Compress-Archive
powershell -Command "Compress-Archive -Path angle_solver.exe,speed_calc.exe -DestinationPath '%ZIPNAME%'"

REM Fallback: if PowerShell doesn't exist or fails, try with 7-Zip if installed
if not exist "%ZIPNAME%" (
    if exist "%ProgramFiles%\7-Zip\7z.exe" (
        "%ProgramFiles%\7-Zip\7z.exe" a "%ZIPNAME%" angle_solver.exe speed_calc.exe
    ) else (
        echo Could not create zip file. Please install 7-Zip or use a system with PowerShell.
        pause
    )
)

REM Done!
echo Packaging complete. Created: %DISTDIR%\%ZIPNAME%
pause
