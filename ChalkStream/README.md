# ChalkStream

## Run Instructions

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you want equation OCR or TrOCR endpoint support, start the OCR server separately and make sure its URL matches the values in [chalkstream_settings.toml](/c:/Users/camer/Documents/00Year4_WorkProducts/ChalkStream-Capstone-Group62/ChalkStream/chalkstream_settings.toml).

Run the application:

```powershell
python main.py
```

The startup screen lets you:

- select a video
- start a live camera recording
- open a saved board
