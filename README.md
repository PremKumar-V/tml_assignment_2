# TML Assignment 2

Trustworthy Machine Learning assignment #2 code repository.

## Setup

### macOS / Linux

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (CMD)

```cmd
python -m venv env
.\env\Scripts\activate.bat
pip install -r requirements.txt
```

## Run

```bash
python huggingface.py
python main.py
```

## Notes

- Run these commands from the repository root (`tml_assignment_2`).
- On macOS Apple Silicon, use `python3` if `python` is unavailable.
- The script automatically selects the best available device (CUDA, MPS, or CPU).
