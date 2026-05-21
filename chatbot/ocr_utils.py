import requests
from PIL import Image, ImageOps
from io import BytesIO
import pytesseract
import re

def extract_text_from_image(url: str) -> str:
    """
    Downloads an image, pre-processes it for better OCR, 
    and extracts text using Tesseract.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 1. Fetch image with browser headers
        r = requests.get(url, headers=headers, timeout=7)
        r.raise_for_status()
        
        # 2. Open and Pre-process
        img = Image.open(BytesIO(r.content))
        
        # Convert to RGB if it's RGBA (transparency can mess up OCR)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Convert to grayscale for better contrast
        img = ImageOps.grayscale(img)
        
        # 3. Extract text with specific configuration
        # --psm 6: Assume a single uniform block of text
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        # 4. Clean specific OCR artifacts and junk characters
        clean_text = re.sub(r'[|~_\[\]@#$]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text) # Remove multiple spaces/newlines
        
        return clean_text.strip()
        
    except Exception as e:
        # Log error for debugging if needed
        # print(f"OCR Error for {url}: {e}")
        return ""