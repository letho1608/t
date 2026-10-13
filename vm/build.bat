@echo off
REM build.bat — build trun.exe bằng MSVC + bootstrap tcomp.tb/trepl.tb (Windows x64).
REM   vm\build.bat            (chạy được từ bất kỳ thư mục nào)
REM Cần Visual Studio Build Tools 2019+ (cl.exe). Nếu đã ở trong "x64 Native
REM Tools Command Prompt" thì dùng luôn cl hiện có.
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"
where cl >nul 2>nul
if not errorlevel 1 goto havecl
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
)
if defined VSDIR (
  call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
) else (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)
REM vcvars có thể đổi thư mục hiện tại
cd /d "%ROOT%"
:havecl
where cl >nul 2>nul
if errorlevel 1 (
  echo [build] khong tim thay cl.exe ^(MSVC^)
  exit /b 1
)
if not exist build mkdir build
cl /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /Fobuild\ /Fe"%ROOT%\trun.exe" vm\tvm.c vm\jit_x64.c vm\tdbg.c vm\trun.c
if errorlevel 1 exit /b 1
echo [build] OK: trun.exe
where python >nul 2>nul
if errorlevel 1 (
  echo [build] khong co python: bo qua bootstrap tcomp.tb
  exit /b 0
)
python vm\tcomp.py -b self\tcli.t tcomp.tb >nul
if errorlevel 1 exit /b 1
"%ROOT%\trun.exe" tcomp.tb -b tools\repl.t trepl.tb >nul
if errorlevel 1 exit /b 1
echo [build] OK: tcomp.tb + trepl.tb
exit /b 0
