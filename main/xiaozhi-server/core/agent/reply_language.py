"""An explicit response-script preference, separate from interface localization."""
LANGUAGES = ('', 'zh-Hans', 'zh-Hant')
RULES = {
    'zh-Hans': 'Reply in Simplified Chinese, unless the user explicitly requests another language. Preserve code and proper names.',
    'zh-Hant': 'Reply in Traditional Chinese (繁體中文), unless the user explicitly requests another language. Preserve code and proper names.',
}


def apply(prompt, language):
    rule = RULES.get(language)
    return prompt + '\n\n[Reply language preference]\n' + rule if rule else prompt
