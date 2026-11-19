"""Проверки формата без печати или возврата секретов."""
import base64
import binascii


def valid_gigachat_authorization_key(value):
    if not isinstance(value,str) or not value:
        return False
    try:
        decoded=base64.b64decode(value,validate=True).decode("utf-8")
    except (ValueError,UnicodeError,binascii.Error):
        return False
    parts=decoded.split(":")
    return len(parts)==2 and all(parts) and not any(c in decoded for c in "\r\n\x00")
