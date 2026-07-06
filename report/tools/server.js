// Static dev server for the report/ directory plus a POST /__shot endpoint
// that writes {name, dataURL} bodies to report/figures/<name>.png.
// (Sandboxed `python3 -m http.server` fails on this machine; see build notes.)
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..'); // report/
const PORT = 8811;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.png': 'image/png',
  '.pdf': 'application/pdf',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.md': 'text/plain; charset=utf-8',
};

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/__shot') {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      try {
        const { name, dataURL } = JSON.parse(body);
        const safe = path.basename(String(name)).replace(/[^\w.-]/g, '_');
        const b64 = String(dataURL).replace(/^data:image\/png;base64,/, '');
        const out = path.join(ROOT, 'figures', safe.endsWith('.png') ? safe : safe + '.png');
        fs.writeFileSync(out, Buffer.from(b64, 'base64'));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, out }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: String(e) }));
      }
    });
    return;
  }
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  let file = path.normalize(path.join(ROOT, urlPath === '/' ? '/_nb/notebook.html' : urlPath));
  if (!file.startsWith(ROOT)) {
    res.writeHead(403);
    return res.end('forbidden');
  }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404);
      return res.end('not found: ' + urlPath);
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

server.listen(PORT, () => console.log(`report server on http://localhost:${PORT}/ root=${ROOT}`));
