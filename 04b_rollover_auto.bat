@echo off
rem Launcher for the "TMS rollover close" scheduled task.
cd /d "%~dp0"
"C:\Users\Asus\AppData\Local\Programs\Python\Python38\python.exe" 04b_rollover_auto.py %*
