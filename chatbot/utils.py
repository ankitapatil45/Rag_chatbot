import base64
import json

def encode_data(data_dict: dict) -> str:
    """
    Standard encoding for the documentation API.
    Converts dictionary -> JSON string -> Base64 string.
    """
    try:
        json_string = json.dumps(data_dict)
        # Encode to bytes, then to base64, then back to a string
        encoded_bytes = base64.b64encode(json_string.encode('utf-8'))
        return encoded_bytes.decode('utf-8')
    except Exception as e:
        print(f"Encoding Error: {e}")
        return ""