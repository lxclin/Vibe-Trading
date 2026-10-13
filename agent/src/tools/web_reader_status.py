"""Recognize reader metadata for an unavailable target, not article keywords."""
import re


def target_page_failure(title: str, text: str):
    warning = re.search(r"^Warning:\s*Target URL returned error\s+(\d{3})\b", text, re.M | re.I)
    if warning and int(warning.group(1)) >= 400:
        return int(warning.group(1)), "target_http_error"
    title = title.strip()
    code = re.fullmatch(r"(404|410|403|429|503)(?:\s*[-:|]\s*[^\n]+)?", title)
    if code:
        return int(code.group(1)), "target_error_page"
    if re.fullmatch(r"(?:page not found|webpage moved|access denied|service unavailable|"
                    r"页面不存在|页面已迁移|访问被拒绝)[.!。！]?", title, re.I):
        return None, "target_unavailable_page"
    return None
