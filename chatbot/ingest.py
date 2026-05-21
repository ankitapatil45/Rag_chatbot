import requests
import re
from chatbot.utils import encode_data
from chatbot.ocr_utils import extract_text_from_image

API_URL = "https://preprod.vmedulife.com/api/helpDesk/documentationPublicData.php"

IMG_REGEX = re.compile(
    r'(https?://[^"\'> ]+\.(png|jpg|jpeg))',
    re.IGNORECASE
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def extract_ocr_from_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    ocr_parts = []
    for url, _ in IMG_REGEX.findall(raw_text):
        ocr = extract_text_from_image(url)
        if ocr:
            ocr_parts.append(ocr)

    return " ".join(ocr_parts)


# ======================================================
# MAIN INGEST FUNCTION
# ======================================================
def collect_documents():
    final_docs = []

    payload = {
        "GetPMInstituteModuleList": "true",
        "data": encode_data({})
    }

    print("📡 Connecting to API for Module List...")

    try:
        response = requests.post(API_URL, data=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return []

    modules_data = data.get("data", {})
    print(f"🔍 Found {len(modules_data)} Root Modules")

    for _, module in modules_data.items():
        module_name = module.get("module_name", "Unknown")
        module_id = module.get("moduleId")

        try:
            p = {
                "HelpDeskGetModuleWiseDemoPointList": "true",
                "data": encode_data({"moduleId": module_id})
            }
            res = requests.post(API_URL, data=p, timeout=15).json()
            deep_data = res.get("data", {})
        except:
            continue

        # ------------------------------------------
        # CRAWLER
        # ------------------------------------------
        def crawl(obj, path):
            if isinstance(obj, dict):
                for v in obj.values():

                    if isinstance(v, (dict, list)):
                        crawl(v, path)

                    elif isinstance(v, str) and len(v.strip()) > 20:
                        cleaned = clean_text(v)
                        ocr_text = extract_ocr_from_text(v)

                        if cleaned or ocr_text:
                            final_docs.append({
                                "text": (
                                    f"LOCATION: {path}\n"
                                    f"TEXT_CONTENT:\n{cleaned}\n\n"
                                    f"OCR_CONTENT:\n{ocr_text}"
                                ),
                                "module": module_name,
                                "path": path,
                                "has_ocr": bool(ocr_text)
                            })

            elif isinstance(obj, list):
                for item in obj:
                    crawl(item, path)

        # 🔴 CRITICAL CALL
        crawl(deep_data, module_name)

    print(f"✅ Total points collected (text + OCR): {len(final_docs)}")
    return final_docs
