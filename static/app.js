/* 前端逻辑:调 POST /api/ask,渲染对话 + 引用来源 */
"use strict";

// 本地直连 FastAPI(8000)与 nginx 反代(80)都用同一路径 /api/ask
const API = "/api";

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newBtn = document.getElementById("new-session");

// 隐私/禁用存储模式会抛 SecurityError → 降级为内存态,不崩脚本
let sessionId;
try {
  sessionId = localStorage.getItem("sid") || newSessionId();
  localStorage.setItem("sid", sessionId);
} catch (e) {
  sessionId = newSessionId();
}

function newSessionId() {
  return "s" + Date.now() + Math.random().toString(36).slice(2, 8);
}

function addMsg(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  wrap.appendChild(body);

  if (sources && sources.length) {
    const box = document.createElement("div");
    box.className = "msg-sources";
    sources.forEach((s) => {
      const tag = document.createElement("span");
      tag.className = "src-tag";
      tag.title = s.file + " 第 " + s.block + " 块";
      tag.textContent = s.file + " #" + s.block;
      box.appendChild(tag);
    });
    wrap.appendChild(box);
  }

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function extractSources(answer) {
  const seen = [];
  const set = new Set();
  const re = /id=([^\s,，;；]+?)#(\d+)/g;
  let m;
  while ((m = re.exec(answer))) {
    if (!set.has(m[1])) {
      set.add(m[1]);
      seen.push({ file: m[1], block: m[2] });
    }
  }
  return seen;
}

async function ask() {
  const q = inputEl.value.trim();
  if (!q || sendBtn.disabled) return;
  inputEl.value = "";
  addMsg("user", q);
  addMsg("bot", "…正在检索并回答,可能需要 10~20 秒");
  sendBtn.disabled = true;

  try {
    const resp = await fetch(API + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, session_id: sessionId }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || ("HTTP " + resp.status));

    chatEl.lastChild.remove(); // 去掉"正在回答"占位
    addMsg("bot", data.answer, extractSources(data.answer));
  } catch (e) {
    chatEl.lastChild.remove();
    addMsg("bot", "❌ 出错了: " + e.message);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

function resetSession() {
  sessionId = newSessionId();
  try { localStorage.setItem("sid", sessionId); } catch (e) { /* 存储不可用则忽略 */ }
  chatEl.innerHTML = "";
}

sendBtn.addEventListener("click", ask);
newBtn.addEventListener("click", resetSession);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
