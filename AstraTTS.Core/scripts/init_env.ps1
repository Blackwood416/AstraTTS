# Create virtual environment for model conversion
Write-Host "--- Creating Python Virtual Environment ---" -ForegroundColor Cyan
python -m venv venv

Write-Host "--- Activating Environment and Installing Dependencies ---" -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch numpy onnx onnxsim onnxruntime

Write-Host "--- Done! ---" -ForegroundColor Green
Write-Host "You can now run the converter using: .\venv\Scripts\python.exe v1_converter.py"
