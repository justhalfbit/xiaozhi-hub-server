/* Translate application-owned copy only. Conversation text, SSIDs and user input are never converted. */
window.UII18n = (() => {
  const key = 'thinksmart.uiLocale';
  const normalize = value => /^zh-(?:Hant|TW|HK|MO)(?:-|$)/i.test(value || '') ? 'zh-Hant' : 'zh-Hans';
  let saved;
  try { saved = localStorage.getItem(key); } catch (_) { /* Private browsing may disable storage. */ }
  const requested = new URLSearchParams(location.search).get('lang');
  const locale = saved === 'zh-Hant' || saved === 'zh-Hans' ? saved
    : requested === 'zh-Hant' || requested === 'zh-Hans' ? requested : normalize(navigator.language);
  if (!saved && requested === locale) {
    try { localStorage.setItem(key, locale); } catch (_) { /* The URL still supplies this visit's locale. */ }
  }
  const catalog = (locale === 'zh-Hant' ? window.UI_ZH_HANT : window.UI_ZH_HANS) || {};
  const t = text => Object.prototype.hasOwnProperty.call(catalog, text) ? catalog[text] : text;
  function init() {
    document.documentElement.lang = locale;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return node.parentElement.closest('script,style,textarea,[data-no-i18n]') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      },
    });
    let node;
    while ((node = walker.nextNode())) {
      const text = node.textContent.trim();
      if (text) node.textContent = node.textContent.replace(text, t(text));
    }
    document.title = t(document.title);
    document.querySelectorAll('[title],[aria-label],[placeholder],[data-kb-enter]').forEach(el => {
      for (const attr of ['title', 'aria-label', 'placeholder', 'data-kb-enter']) {
        if (el.hasAttribute(attr)) el.setAttribute(attr, t(el.getAttribute(attr)));
      }
    });
    document.querySelectorAll('[data-language-picker]').forEach(select => {
      select.value = locale;
      select.addEventListener('change', () => {
        try { localStorage.setItem(key, select.value); } catch (_) { return; }
        location.reload();
      });
    });
  }
  return { t, init, locale, normalize };
})();
