@echo off
REM T language one-click installer: chép dự án vào D:\Code\t, build VM C, chạy toàn bộ test.
REM   INSTALL_TO_D.bat            (chạy từ thư mục nguồn bất kỳ)
setlocal
set "SRC=%~dp0"
set "DST=D:\Code\t"
if /I "%SRC%"=="%DST%\" (
  echo [T] nguon trung dich, bo qua buoc chep
  goto :build
)
for %%D in (std tests bench self tools vm .github\workflows) do mkdir "%DST%\%%D" 2>nul
for %%F in (t.py ttensor.py SPEC.md README.md STATUS.md TODO.md) do xcopy "%SRC%%%F" "%DST%\" /Y /Q >nul
xcopy "%SRC%std\*.t" "%DST%\std\" /Y /Q >nul
xcopy "%SRC%tests\*.t" "%DST%\tests\" /Y /Q >nul
xcopy "%SRC%bench\*" "%DST%\bench\" /Y /Q >nul
xcopy "%SRC%self\*.t" "%DST%\self\" /Y /Q >nul
xcopy "%SRC%tools\*.t" "%DST%\tools\" /Y /Q >nul
xcopy "%SRC%tools\*.py" "%DST%\tools\" /Y /Q >nul
xcopy "%SRC%vm\*" "%DST%\vm\" /Y /Q >nul
xcopy "%SRC%.github\workflows\*" "%DST%\.github\workflows\" /Y /Q >nul
echo [T] copied to %DST%

:build
echo.
echo [T] interpreter demo (python t.py)...
python "%DST%\t.py" "%DST%\tests\demo.t"
if errorlevel 1 goto :fail
echo.
echo [T] building C VM (MSVC) + bootstrap tcomp.tb/trepl.tb...
call "%DST%\vm\build.bat"
if errorlevel 1 (
  echo [T] khong build duoc trun.exe ^(can MSVC^) - chi co interpreter t.py
  goto :fail
)
echo.
echo [T] running ALL tests (tools\run_all.py)...
pushd "%DST%"
python tools\run_all.py
set RC=%errorlevel%
popd
if not "%RC%"=="0" goto :fail
echo.
echo [T] ALL GREEN
pause
exit /b 0
:fail
echo [T] FAILED
pause
exit /b 1
