@echo off
REM T language one-click installer: copy project into D:\Code\t, build C VM, run full test suite.
REM   INSTALL_TO_D.bat            (runnable from any source directory)
setlocal
set "SRC=%~dp0"
set "DST=D:\Code\t"
if /I "%SRC%"=="%DST%\" (
  echo [T] Source is destination, skipping copy step
  goto :build
)
for %%D in (std tests bench self tools vm .github\workflows) do mkdir "%DST%\%%D" 2>nul
for %%F in (tcomp.tb trepl.tb SPEC.md README.md STATUS.md TODO.md) do if exist "%SRC%%%F" xcopy "%SRC%%%F" "%DST%\" /Y /Q >nul
xcopy "%SRC%std\*.t" "%DST%\std\" /Y /Q >nul
xcopy "%SRC%tests\*.t" "%DST%\tests\" /Y /Q >nul
xcopy "%SRC%bench\*" "%DST%\bench\" /Y /Q >nul
xcopy "%SRC%self\*.t" "%DST%\self\" /Y /Q >nul
xcopy "%SRC%tools\*.t" "%DST%\tools\" /Y /Q >nul
xcopy "%SRC%vm\*" "%DST%\vm\" /Y /Q >nul
xcopy "%SRC%.github\workflows\*" "%DST%\.github\workflows\" /Y /Q >nul
echo [T] copied to %DST%

:build
echo.
echo [T] building C VM with MSVC and self-hosting tcomp.tb/trepl.tb...
call "%DST%\vm\build.bat"
if errorlevel 1 (
  echo [T] failed to build trun.exe - MSVC required
  goto :fail
)
echo.
echo [T] running ALL tests with trun tools\run_all.t...
pushd "%DST%"
.\trun.exe tools\run_all.t
set RC=%errorlevel%
popd
if not "%RC%"=="0" goto :fail
echo.
echo [T] ALL GREEN
exit /b 0
:fail
echo [T] FAILED
exit /b 1
