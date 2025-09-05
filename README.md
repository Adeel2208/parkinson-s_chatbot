```markdown
# Parkinson's Disease Chatbot

A production-ready chatbot providing Parkinson's disease insights from PDF data, optimized for Kaggle.

## Prerequisites
- Python 3.10+
- Kaggle environment or local setup with libraries from [requirements.txt](#requirements)
- Hugging Face token (set as `HF_TOKEN` environment variable)

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Adeel2208/parkinson-s_chatbot.git
   cd parkinson-s_chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```
Dataset Link: https://drive.google.com/file/d/1MeFV4oyNHouLPuI-KKTIPfHo1cccZHDN/view?usp=drive_link
   
3. Install the package:
   ```bash
   python setup.py install
   ```
4. (Optional) Set `HF_TOKEN` in a `.env` file or environment (e.g., `export HF_TOKEN=your_token`).

## Usage
### On Kaggle
1. Upload the `parkinson-s_chatbot` directory to your Kaggle notebook.
2. Ensure PDFs are at `/kaggle/input/parkinsons-disease2455/dataset/training/parkinson disease`.
3. Run files in order via notebook cells:
   - `src/config/settings.py`
   - `src/data/processing.py`
   - `src/embedding/store.py`
   - `src/agents/graph.py`
   - `src/interface/chat.py`
4. Type questions (e.g., "What exercises help with Parkinson’s tremors?") and press Enter. Type `exit` to quit.

### Locally
1. Adjust `PDF_FOLDER` in `src/config/settings.py` if needed.
2. Run:
   ```bash
   parkinsons-chat
   ```

## File Structure
- `src/`: Core Python code
- `tests/`: Unit tests
- `docs/`: Documentation
- `setup.py`: Installation script

## Requirements
Install dependencies from [docs/requirements.txt](docs/requirements.txt).

## Contributing
1. Fork the repo.
2. Create a branch: `git checkout -b feature-branch`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push and submit a PR.

## License
[MIT](LICENSE)
```
