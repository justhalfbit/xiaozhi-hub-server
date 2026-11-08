"""An explicit response-script preference, separate from interface localization."""
OPTIONS = (
    {'value': '', 'label': '自動／沿用模型設定'},
    {'value': 'zh-Hant', 'label': '國語（繁體中文）'},
    {'value': 'zh-Hans', 'label': '普通話（簡體中文）'},
    {'value': 'yue-HK', 'label': '廣東話／粵語（香港）'},
    {'value': 'en', 'label': '英語 English'},
    {'value': 'ja', 'label': '日語 日本語'},
    {'value': 'ko', 'label': '韓語 한국어'},
    {'value': 'es', 'label': '西班牙語 Español'},
    {'value': 'fr', 'label': '法語 Français'},
)
LANGUAGES = tuple(option['value'] for option in OPTIONS)
RULES = {
    'zh-Hans': 'Reply in Mandarin Chinese using Simplified Chinese characters, unless the user explicitly requests another language. Preserve code and proper names.',
    'zh-Hant': 'Reply in Traditional Chinese (繁體中文), using natural Mandarin wording, unless the user explicitly requests another language. Preserve code and proper names.',
    'yue-HK': 'Reply in natural colloquial Hong Kong Cantonese (廣東話／粵語), written in Traditional Chinese. Use Cantonese wording such as 係、唔、嘅、喺、冇 rather than written Mandarin. Preserve code and proper names. Follow an explicit user request for another language.',
    'en': 'Reply in English unless the user explicitly requests another language. Preserve code and proper names.',
    'ja': 'Reply in Japanese (日本語) unless the user explicitly requests another language. Preserve code and proper names.',
    'ko': 'Reply in Korean (한국어) unless the user explicitly requests another language. Preserve code and proper names.',
    'es': 'Reply in Spanish (Español) unless the user explicitly requests another language. Preserve code and proper names.',
    'fr': 'Reply in French (Français) unless the user explicitly requests another language. Preserve code and proper names.',
}

PHRASES = {
    '': {'wake': '在', 'call': '我在', 'test': '你好，這是語音測試。', 'not_ready': '还没配置大模型，请在电脑上打开网关后台填好 API Key。'},
    'zh-Hant': {'wake': '在', 'call': '我在', 'test': '你好，這是國語語音測試。', 'not_ready': '還沒設定大語言模型，請在電腦上開啟網關管理頁填好 API Key。'},
    'zh-Hans': {'wake': '在', 'call': '我在', 'test': '你好，这是普通话语音测试。', 'not_ready': '还没配置大模型，请在电脑上打开网关后台填好 API Key。'},
    'yue-HK': {'wake': '喺度', 'call': '我喺度', 'test': '你好，呢個係廣東話語音測試。', 'not_ready': '仲未設定語言模型，請喺電腦開網關管理頁，填好 API Key。'},
    'en': {'wake': "I'm here.", 'call': "I'm here.", 'test': 'Hello, this is an English speech test.', 'not_ready': 'No language model is configured. Open the gateway admin page on your computer and enter an API key.'},
    'ja': {'wake': 'はい', 'call': 'はい、聞いています。', 'test': 'こんにちは。これは日本語の音声テストです。', 'not_ready': '言語モデルが設定されていません。パソコンでゲートウェイ管理画面を開き、API キーを入力してください。'},
    'ko': {'wake': '네', 'call': '네, 듣고 있어요.', 'test': '안녕하세요. 한국어 음성 테스트입니다.', 'not_ready': '언어 모델이 설정되지 않았습니다. 컴퓨터에서 게이트웨이 관리 페이지를 열고 API 키를 입력하세요.'},
    'es': {'wake': 'Aquí estoy.', 'call': 'Aquí estoy.', 'test': 'Hola, esta es una prueba de voz en español.', 'not_ready': 'No hay un modelo de lenguaje configurado. Abre la página de administración e introduce una clave API.'},
    'fr': {'wake': 'Je suis là.', 'call': 'Je suis là.', 'test': 'Bonjour, ceci est un test vocal en français.', 'not_ready': "Aucun modèle de langue n'est configuré. Ouvrez la page d'administration et saisissez une clé API."},
}


def phrase(kind, language=''):
    return PHRASES.get(language, PHRASES[''])[kind]


def apply(prompt, language):
    rule = RULES.get(language)
    return prompt + '\n\n[Reply language preference]\n' + rule if rule else prompt
