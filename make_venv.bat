@echo off
REM Python 가상환경 생성 및 활성화
py -3.10 -m venv venv
call venv/Scripts/activate

REM 필요한 패키지 설치
cmd /c "pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128"
python -m pip install -r requirements.txt

echo Installation completed! You can now run ComfyUI using run_gpu.bat
pause