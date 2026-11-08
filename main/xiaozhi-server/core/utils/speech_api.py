"""Speech response helpers; never expose complete credentials in provider errors."""
import re


def response_error(response, api_key, label):
    message = ''
    try:
        body = response.json()
        error = body.get('error') or {}
        if isinstance(error, dict):
            message = str(error.get('message') or '')
    except (ValueError, TypeError, AttributeError):
        pass
    if api_key:
        message = message.replace(api_key, '[REDACTED]')
    message = re.sub(r'(?i)bearer\s+\S+|sk-[\w-]+', '[REDACTED]', message)
    message = ' '.join(message.split())[:300]
    return RuntimeError(f'{label}: HTTP {response.status_code}' + (f' — {message}' if message else ''))
