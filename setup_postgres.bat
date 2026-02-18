@echo off
REM PostgreSQL Setup Script for Windows

echo ========================================================================
echo PostgreSQL Setup for CCTV People Counting System (Windows)
echo ========================================================================
echo.

echo Choose setup method:
echo   1. Docker (Easiest - Recommended)
echo   2. Manual PostgreSQL Installation Guide
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" goto docker
if "%choice%"=="2" goto manual
echo Invalid choice
goto end

:docker
echo.
echo ========================================================================
echo Docker Setup
echo ========================================================================
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo X Docker not found
    echo.
    echo Please install Docker Desktop for Windows:
    echo   https://www.docker.com/products/docker-desktop
    echo.
    pause
    goto end
)

echo + Docker found
echo.

REM Check if docker-compose is installed
where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo X docker-compose not found
    echo   Should be included with Docker Desktop
    pause
    goto end
)

echo + docker-compose found
echo.

REM Create .env file
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo + .env file created
) else (
    echo + .env file already exists
)

echo.
echo Starting PostgreSQL with Docker...
docker-compose up -d

echo.
echo Waiting for PostgreSQL to be ready...
timeout /t 5 /nobreak >nul

echo.
echo === PostgreSQL is running! ===
echo.
echo Connection details:
echo   Host: localhost
echo   Port: 5432
echo   Database: cctv_counting
echo   User: ferbos
echo   Password: cctv_ferbos_2024
echo.
echo To stop PostgreSQL:
echo   docker-compose down
echo.
echo To view logs:
echo   docker-compose logs -f

goto next_steps

:manual
echo.
echo ========================================================================
echo Manual PostgreSQL Installation
echo ========================================================================
echo.
echo Please follow these steps:
echo.
echo 1. Download PostgreSQL from:
echo    https://www.postgresql.org/download/windows/
echo.
echo 2. Run the installer
echo    - Remember the password you set for 'postgres' user
echo.
echo 3. After installation, create database:
echo    a. Open SQL Shell (psql)
echo    b. Login as postgres user
echo    c. Run: CREATE DATABASE cctv_counting;
echo.
echo 4. Create .env file with your settings:
echo    DB_HOST=localhost
echo    DB_PORT=5432
echo    DB_NAME=cctv_counting
echo    DB_USER=postgres
echo    DB_PASSWORD=your_password
echo.
pause
goto next_steps

:next_steps
echo.
echo ========================================================================
echo Next Steps
echo ========================================================================
echo.
echo 1. Install Python dependencies:
echo    pip install -r requirements.txt
echo.
echo 2. (Optional) Migrate existing JSON data:
echo    python migrate_json_to_postgres.py
echo.
echo 3. Run the application:
echo    python main.py
echo.
echo 4. For more information:
echo    See POSTGRESQL_SETUP.md
echo.
echo ========================================================================

:end
pause