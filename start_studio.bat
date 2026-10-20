@echo off
title Dewatermark AI Studio
cd /d "%~dp0"

echo =======================================================================
echo    Dewatermark AI Studio - Precision Watermark Remover
echo =======================================================================
echo  [*] Menyiapkan server lokal di http://127.0.0.1:5000 ...
echo  [*] Browser akan otomatis terbuka dalam 2 detik...
echo.
echo  [!] PENTING: Biarkan jendela ini tetap terbuka selama menggunakan Studio.
echo      Tutup jendela ini jika Anda sudah selesai.
echo =======================================================================
echo.

:: Otomatis buka web browser ke localhost:5000
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Jalankan server Flask
python app.py

echo.
echo Server telah berhenti.
pause
