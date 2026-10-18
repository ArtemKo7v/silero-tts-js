// Run the real demo in an isolated headless Chrome profile using its built-in CDP endpoint.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const input = process.argv[2];
if (!input) throw new Error("Usage: node tests/smoke-browser.mjs URL_OR_HTML_PATH");
const url = new URL(/^https?:|^file:/.test(input) ? input : pathToFileURL(path.resolve(input)).href);
url.hash = "autotest";
const candidates = [process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"];
const executable = candidates.find((p) => p && existsSync(p));
if (!executable) throw new Error("Chrome not found; set CHROME_PATH");
const work = path.join(root, ".work");
await mkdir(work, { recursive: true });
const profile = await mkdtemp(path.join(work, "chrome-"));
const browser = spawn(executable, ["--headless=new", "--remote-debugging-port=0",
  `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "about:blank"],
{ windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
let socket;
const pending = new Map();
let nextId = 0;
const errors = [];
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = "";
    const timer = setTimeout(() => reject(new Error("Chrome endpoint timeout")), 30000);
    browser.on("error", (error) => { clearTimeout(timer); reject(error); });
    browser.on("exit", (code) => { clearTimeout(timer); reject(new Error(`Chrome exited: ${code}`)); });
    browser.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  socket = new WebSocket(endpoint);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  socket.addEventListener("message", ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) {
      const waiter = pending.get(message.id);
      if (!waiter) return;
      pending.delete(message.id);
      clearTimeout(waiter.timer);
      if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
      else waiter.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") {
      errors.push(message.params.exceptionDetails);
    }
  });
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const call = (method, params = {}) => send(method, params, sessionId);
  const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call("Runtime.enable");
  await call("Page.enable");
  await call("Page.navigate", { url: url.href });
  const deadline = Date.now() + 240000;
  let title = "", lastStatus = "";
  while (Date.now() < deadline) {
    const state = await evaluate("({title:document.title,status:document.getElementById('status')?.textContent})");
    title = state.title;
    if (state.status && state.status !== lastStatus) {
      console.log(state.status);
      lastStatus = state.status;
    }
    if (title.startsWith("AUTOTEST ")) break;
    await delay(1000);
  }
  assert.ok(title.startsWith("AUTOTEST {"), `Autotest failed or timed out: ${title}; ${JSON.stringify(errors)}`);
  const result = JSON.parse(title.slice("AUTOTEST ".length));
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.audio, true);
  assert.ok(result.synthesis.samples > 0);
  if (url.protocol === "file:") assert.equal(result.isolated, false);
  const wav = await evaluate(`(async () => {
    const response = await fetch(document.getElementById('audio').src);
    const buffer = await response.arrayBuffer(), view = new DataView(buffer);
    const tag = (start) => String.fromCharCode(...new Uint8Array(buffer, start, 4));
    let peak = 0;
    for (let i = 44; i < buffer.byteLength; i += 2) peak = Math.max(peak, Math.abs(view.getInt16(i, true)));
    return {riff:tag(0), wave:tag(8), bytes:buffer.byteLength, channels:view.getUint16(22,true),
      rate:view.getUint32(24,true), bits:view.getUint16(34,true), dataBytes:view.getUint32(40,true), peak};
  })()`);
  assert.equal(wav.riff, "RIFF");
  assert.equal(wav.wave, "WAVE");
  assert.equal(wav.channels, 1);
  assert.equal(wav.rate, 24000);
  assert.equal(wav.bits, 16);
  assert.equal(wav.dataBytes, result.synthesis.samples * 2);
  assert.equal(wav.bytes, wav.dataBytes + 44);
  assert.ok(wav.peak > 100, "Synthesized PCM must not be silent");
  const name = url.protocol === "file:" ? "offline" : "web";
  const screenshot = await call("Page.captureScreenshot", { format: "png" });
  await writeFile(path.join(work, `${name}.png`), Buffer.from(screenshot.data, "base64"));
  const report = { ...result, wav };
  await writeFile(path.join(work, `${name}-smoke.json`), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report));
  await send("Browser.close");
} finally {
  socket?.close();
  for (const waiter of pending.values()) clearTimeout(waiter.timer);
  if (browser.exitCode === null) {
    const exited = new Promise((resolve) => browser.once("exit", resolve));
    browser.kill();
    await Promise.race([exited, delay(5000)]);
  }
  // Remove only this test's freshly created profile inside the workspace.
  assert.equal(path.dirname(profile), work);
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
}
