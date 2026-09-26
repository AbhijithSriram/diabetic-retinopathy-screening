/* DrishtiTrust shared helpers: outcome shapes, toasts, JSON fetch, read aloud.
   No framework; every page adds its own small script. */
(function () {
  const SHAPE = {
    recall: '<circle cx="8" cy="8" r="7" fill="currentColor"/>',
    review: '<polygon points="8,0.5 15.5,8 8,15.5 0.5,8" fill="currentColor"/>',
    refer: '<polygon points="8,1 15.5,15 0.5,15" fill="currentColor"/>',
    urgent: '<polygon points="8,1 15.5,15 0.5,15" fill="currentColor"/>',
    ungradable: '<rect x="1.5" y="1.5" width="13" height="13" fill="currentColor"/>',
    info: '<circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.8"/>',
  };

  const DT = {
    shape(kind, size = 14) {
      return `<svg width="${size}" height="${size}" viewBox="0 0 16 16" aria-hidden="true">${SHAPE[kind] || SHAPE.info}</svg>`;
    },
    esc(s) {
      return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },
    async json(url, opts = {}) {
      const r = await fetch(url, opts);
      if (!r.ok) {
        const err = new Error(`${r.status}`);
        err.status = r.status;
        try { err.detail = (await r.json()).detail; } catch (e) { /* not JSON */ }
        throw err;
      }
      return r.json();
    },
    post(url, body) {
      return DT.json(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    },
    toast(text, action) {
      const box = document.getElementById('toasts');
      const t = document.createElement('div');
      t.className = 'toast';
      t.innerHTML = `<span style="flex:1">${DT.esc(text)}</span>`;
      const b = document.createElement('button');
      b.className = 'toast__btn';
      b.textContent = action ? action.label : 'OK';
      b.onclick = () => { if (action) action.run(); t.remove(); };
      t.appendChild(b);
      box.appendChild(t);
      setTimeout(() => t.remove(), 6000);
    },
    hm(t) { return new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }); },
    dmy(t) { return new Date(t * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }); },
    day(s) {  /* "2026-10-07" -> "Wed 7 Oct 2026" */
      if (!s) return '';
      const d = new Date(s + 'T00:00:00');
      return isNaN(d) ? s : d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
    },
    isoDay(offsetDays) {
      const d = new Date(); d.setDate(d.getDate() + offsetDays);
      return d.toISOString().slice(0, 10);
    },
    /* read aloud with the browser's voice for the language; hides itself when no voice exists */
    voiceFor(lang) {
      if (!('speechSynthesis' in window)) return null;
      const want = { en: 'en', hi: 'hi', ta: 'ta' }[lang] || 'en';
      return speechSynthesis.getVoices().find(v => v.lang && v.lang.toLowerCase().startsWith(want)) || null;
    },
    speak(text, lang) {
      if (!('speechSynthesis' in window)) return false;
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = { en: 'en-IN', hi: 'hi-IN', ta: 'ta-IN' }[lang] || 'en-IN';
      const v = DT.voiceFor(lang);
      if (v) u.voice = v;
      u.rate = 0.9;
      speechSynthesis.speak(u);
      return true;
    },
  };
  window.DT = DT;
})();
