python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytesseract pillow
ollama pull tinyllama
python -m chatbot.embed
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4



--Install system dependency
sudo apt update
sudo apt install -y tesseract-ocr
pip install redis
redis-server




rm -rf .chroma
python -m chatbot.embed
pkill uvicorn
uvicorn main:app --workers 4

