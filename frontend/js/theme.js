/* NYC Flood Watch: light / dark theme.
   First visit follows the computer's setting; the moon/sun button in the bar switches it and the choice is remembered.
   Loaded in <head> (not deferred) so the page never flashes the wrong colours. Pages listen for "themechange" to redraw charts. */
"use strict";
(function () {
  const KEY = "fw-theme", sys = window.matchMedia ? matchMedia("(prefers-color-scheme: dark)") : null;
  const saved = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const root = document.documentElement;
  const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';
  const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/></svg>';
  let btn = null;
  const sync = () => { if (!btn) return; const d = root.dataset.theme === "dark";
    btn.innerHTML = d ? SUN : MOON; btn.setAttribute("aria-label", d ? "Switch to light mode" : "Switch to dark mode"); btn.title = btn.getAttribute("aria-label"); };
  function apply(t, announce) {
    root.dataset.theme = t; sync();
    if (announce) window.dispatchEvent(new CustomEvent("themechange", { detail: t }));
  }
  apply(saved() || (sys && sys.matches ? "dark" : "light"), false);
  if (sys && sys.addEventListener) sys.addEventListener("change", e => { if (!saved()) apply(e.matches ? "dark" : "light", true); });
  window.Theme = {
    dark: () => root.dataset.theme === "dark",
    toggle() { const t = this.dark() ? "light" : "dark"; try { localStorage.setItem(KEY, t); } catch (e) {} apply(t, true); }
  };
  document.addEventListener("DOMContentLoaded", () => {
    const nav = document.querySelector(".rail"); if (!nav) return;
    btn = document.createElement("button"); btn.type = "button"; btn.className = "theme-btn"; btn.onclick = () => Theme.toggle();
    nav.appendChild(btn); sync();
  });
})();
