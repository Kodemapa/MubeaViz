@echo off
title HDF5 File Copy Script
color 0A
cls

:: CONFIGURATION
set "INPUT=../input"
set "OUTPUT=../valid"

echo ----------------------------------------
echo Copying NEW HDF5 files only (preserving timestamps)
echo From: %INPUT%
echo To:   %OUTPUT%
echo ----------------------------------------
echo.

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

:: Count files for progress reporting
set "total=0"
for %%F in ("%INPUT%\*.h5") do set /a total+=1
echo Found %total% HDF5 files to process.
echo.

:: Process counter
set "processed=0"

:: LOOP through all h5 files with inline processing
for %%F in ("%INPUT%\*.h5") do (
    set /a processed+=1
    echo [%processed%/%total%] Processing: %%~nxF
    
    if exist "%OUTPUT%\%%~nxF" (
        echo    - Skipping: File already exists
    ) else (
        echo    - Copying file...
        copy "%%F" "%OUTPUT%\" > nul
        echo    - Setting timestamp...
        powershell -Command "$file = Get-Item -Path '%%F'; Set-ItemProperty -Path '%OUTPUT%\%%~nxF' -Name LastWriteTime -Value $file.LastWriteTime" > nul
        echo    - Done
    )
    echo.
)

echo All files processed successfully.
echo.
echo Press any key to exit...
pause > nul