## How to Run the App (Windows)

From the project folder:
`D:\Salman AI\Time-Series-Python-App`

### Option A: Use venv’s Python (no activation needed)
```powershell
.\n+venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port=8501 --server.headless=true
```

### Option B: Activate venv (if your policy allows) and run
```powershell
.
venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py --server.port=8501 --server.headless=true
```

If activation is blocked, either use Option A or this one‑liner that bypasses policy for this session only:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command ". .\venv\Scripts\Activate.ps1; python -m streamlit run streamlit_app.py --server.port=8501 --server.headless=true"
```

### Alternate launcher (installs then runs)
```powershell
python run_app.py
```

### If port 8501 is busy
```powershell
python -m streamlit run streamlit_app.py --server.port=8502 --server.headless=true
```

## First‑Time Setup (if needed)
```powershell
python -m venv venv
.
venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies are pinned to avoid ABI issues:
- `numpy<2.0`
- `torch==2.0.1+cpu` (matches installed torchaudio/torchvision)

## Quick Health Check
```powershell
powershell -NoProfile -Command "(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/ -TimeoutSec 5).StatusCode"
```
Expect `200` when the app is ready.

## Troubleshooting

- Execution policy blocks Activate.ps1:
  - Use Option A (no activation), or:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
  # or unblock just the file
  Unblock-File .\venv\Scripts\Activate.ps1
  ```

- Port 8501 already in use:
  - Run on 8502 (see above), or free the port:
  ```powershell
  $pid=(Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if($pid){ Stop-Process -Id $pid -Force }
  ```

- NumPy ABI error (module compiled for NumPy 1.x):
  - Already mitigated in requirements with `numpy<2.0`.

- Torch/torchaudio resolver warnings:
  - Pinned `torch==2.0.1+cpu` to align with torchaudio/torchvision.


