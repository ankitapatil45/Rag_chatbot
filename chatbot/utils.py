# ============================================================
# utils.py — Shared encoding utilities
# No changes needed from v1 — this file was already clean.
# ============================================================

import base64
import json
import logging

logger = logging.getLogger(__name__)


def encode_data(data_dict: dict) -> str:
    """
    Serialize `data_dict` to JSON, then Base64-encode it.
    Used to build the API payloads for the ERP documentation endpoint.
    """
    try:
        json_string = json.dumps(data_dict)
        encoded_bytes = base64.b64encode(json_string.encode("utf-8"))
        return encoded_bytes.decode("utf-8")
    except Exception as e:
        logger.error("encode_data failed: %s", e)
        return ""