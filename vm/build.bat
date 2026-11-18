@echo off
REM build.bat — build trun.exe with MSVC + self-host tcomp.tb/trepl.tb (Windows x64).
REM   vm\build.bat            (runnable from any directory)
REM Requires Visual Studio Build Tools 2019+ (cl.exe).
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
cd /d "%ROOT%"
:havecl
where cl >nul 2>nul
if errorlevel 1 (
  echo [build] cl.exe (MSVC) not found
  exit /b 1
)
if not exist build mkdir build
cl /nologo /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /Fobuild\ /Fe"%ROOT%\trun.exe" vm\tvm.c vm\jit_x64.c vm\jit_arm64.c vm\tdbg.c vm\trun.c
set BUILD_RC=%errorlevel%
rmdir /s /q build 2>nul
del /q *.obj 2>nul
if not "%BUILD_RC%"=="0" exit /b 1
echo [build] OK: trun.exe
if exist "%ROOT%\tcomp.tb" (
  "%ROOT%\trun.exe" tcomp.tb -b self\tcli.t tcomp.new.tb >nul
  if errorlevel 1 exit /b 1
  move /y tcomp.new.tb tcomp.tb >nul
  "%ROOT%\trun.exe" tcomp.tb -b tools\repl.t trepl.tb >nul
  if errorlevel 1 exit /b 1
  echo [build] OK: tcomp.tb + trepl.tb (self-hosted)
)
exit /b 0
