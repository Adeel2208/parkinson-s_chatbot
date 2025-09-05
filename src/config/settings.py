import os

class Settings:
    PDF_FOLDER = "/kaggle/input/parkinsons-disease2455/dataset/training/parkinson disease"
    PERSIST_DIR = "./chroma_db"
    HF_TOKEN = os.getenv("HF_TOKEN", "hugging_face_token")  # Allow env override
    MAX_NEW_TOKENS = 200
    TEMPERATURE = 0.2

settings = Settings()