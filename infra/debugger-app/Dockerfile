FROM gcr.io/skia-public/debugger-app-base@sha256:16abd26d2ada7e3118b4777917d878058bde44061fa6be14b279a6e8ff3859d8

COPY canvaskit.js /usr/local/share/debugger-app/canvaskit.js
COPY canvaskit.wasm /usr/local/share/debugger-app/canvaskit.wasm
COPY version.js /usr/local/share/debugger-app/version.js
COPY index.d.ts /usr/local/share/debugger-app/index.d.ts

CMD ["/usr/local/bin/debugger", "--help"]