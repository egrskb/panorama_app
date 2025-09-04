autoattach_win.bat@echo off
setlocal enabledelayedexpansion

echo Finding HackRF...
for /f "tokens=1,* delims=:" %%A in ('usbipd list ^| findstr /i "HackRF"') do (
    for /f "tokens=1" %%X in ("%%A") do (
        set "BUSID=%%X"
        echo Disconnecting HackRF with BusID=!BUSID!...
        usbipd detach --busid !BUSID!
    )
)

echo.
echo Done.
pause >nul