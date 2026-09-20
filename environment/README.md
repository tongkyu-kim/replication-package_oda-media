# Environment

- **Python**: 3.10 or later (developed and tested on 3.13.1)
- **OS**: developed on Windows 10/11; no OS-specific code paths are used (all paths are built with `pathlib`), so Linux/macOS should work identically
- **GPU**: optional. Classifier validation (`classifier_validation.py`) uses `torch`/`transformers` and will use a CUDA GPU automatically if available (`torch.cuda.is_available()`), otherwise runs on CPU (slower, not incompatible)

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

No `.env` file, API keys, or network access is required to run this package. All external data this package depends on (World Bank, UN Comtrade, UCDP, EM-DAT pulls) is already cached under `data/analytical/`; see `DATA_AVAILABILITY.md` for provenance.
