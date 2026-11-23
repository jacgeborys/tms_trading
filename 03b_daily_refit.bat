@echo off
rem Launcher for the "TMS daily ladder refit" scheduled task.
cd /d "%~dp0"
"C:\Users\Asus\AppData\Local\Programs\Python\Python38\python.exe" 03b_daily_refit.py %*
