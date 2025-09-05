from setuptools import setup, find_packages

setup(
    name="parkinsons_chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.1.0",
        "langchain-huggingface>=0.0.1",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "PyPDF2>=3.0.0",
        "sentence-transformers>=2.2.0",
        "langgraph>=0.0.4",
    ],
    entry_points={
        "console_scripts": [
            "parkinsons-chat=src.interface.chat:main",
        ],
    },
)