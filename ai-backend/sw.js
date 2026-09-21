/* =========================================================
   AI MOVEMENT COACH — SERVICE WORKER
   Scope: app-shell caching only (index.html, style.css, script.js,
   manifest, icons). Deliberately does NOT intercept:
   - cross-origin requests (MediaPipe CDN, Gemini backend) — those
     must always hit the network directly, never be served stale.
   - any request whose method isn't GET.
   This keeps the "on-device live coaching never depends on this
   service worker" property true — if the SW fails to install or
   is unsupported, the app works exactly the same via network.
========================================================= */

const CACHE_NAME = "movement-coach-shell-v1";
const SHELL_FILES = [
    "./index.html",
    "./style.css",
    "./script.js",
    "./manifest.json",
    "./icons/icon-192.png",
    "./icons/icon-512.png"
];

self.addEventListener("install", event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(SHELL_FILES))
            .catch(error => console.warn("[SW] Shell cache failed (app still works online):", error))
    );
    self.skipWaiting();
});

self.addEventListener("activate", event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))
        )
    );
    self.clients.claim();
});

self.addEventListener("fetch", event => {
    const request = event.request;

    // Only ever handle same-origin GET requests for the app shell.
    // Everything else (MediaPipe CDN, Gemini backend, any POST) goes
    // straight to the network untouched.
    if (request.method !== "GET") return;

    const url = new URL(request.url);
    if (url.origin !== self.location.origin) return;

    event.respondWith(
        caches.match(request).then(cached => {
            if (cached) return cached;
            return fetch(request).catch(() => cached);
        })
    );
});