"""Build a corpus of request bodies (chat / messages / responses) from opencode sessions.

Usage: python corpus_from_opencode.py <path/to/opencode.db> <output dir>

The six largest sessions are written in three request formats each, every
session in a different client serialisation style (compact, ASCII-escaped,
spaced, indented, sorted keys). The output is real conversation content: keep
it out of the repository and point the Go tests at it with AIR_BODY_CORPUS.
"""
import sqlite3, json, os, sys
db, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
c = sqlite3.connect('file:' + db.replace(os.sep, '/') + '?mode=ro', uri=True)
sessions = [r[0] for r in c.execute("select session_id from session_message group by 1 order by sum(length(data)) desc limit 6")]

def load(sid):
    turns = []
    for t, data in c.execute("select type, data from session_message where session_id=? order by seq", (sid,)):
        turns.append((t, json.loads(data)))
    return turns

def tool_out(part):
    st = part.get("state") or {}
    txt = "".join(x.get("text", "") for x in (st.get("content") or []) if isinstance(x, dict))
    return txt or json.dumps(st.get("metadata") or {}, ensure_ascii=False)

def tools_of(turns):
    names = []
    for t, m in turns:
        if t == "assistant":
            for p in m.get("content", []):
                if p.get("type") == "tool" and p.get("name") not in names: names.append(p["name"])
    return names or ["bash"]

def schema(name):
    return {"type": "object", "properties": {"command": {"type": "string", "description": "что выполнить / what to run"},
            "path": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}}, "required": ["command"], "additionalProperties": False}

def chat(turns, model):
    msgs = []
    for t, m in turns:
        if t in ("system",): msgs.append({"role": "system", "content": m.get("text", "")})
        elif t in ("user", "synthetic"): msgs.append({"role": "user", "content": m.get("text", "")})
        elif t == "compaction": msgs.append({"role": "user", "content": "[summary] " + (m.get("summary") or "")})
        elif t == "assistant":
            text, reasoning, calls, results = "", "", [], []
            for p in m.get("content", []):
                if p.get("type") == "text": text += p.get("text", "")
                elif p.get("type") == "reasoning": reasoning += p.get("text", "")
                elif p.get("type") == "tool":
                    calls.append({"id": p["id"], "type": "function", "function": {"name": p["name"], "arguments": json.dumps((p.get("state") or {}).get("input") or {}, ensure_ascii=False)}})
                    results.append({"role": "tool", "tool_call_id": p["id"], "content": tool_out(p)})
            a = {"role": "assistant", "content": text or None}
            if reasoning: a["reasoning_content"] = reasoning
            if calls: a["tool_calls"] = calls
            msgs.append(a); msgs.extend(results)
    return {"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "function": {"name": n, "description": "Tool " + n, "parameters": schema(n)}} for n in tools_of(turns)],
            "tool_choice": "auto", "temperature": 0.2, "max_tokens": 8192, "user": "user-42", "reasoning_effort": "medium"}

def messages(turns, model):
    system, msgs = [], []
    def push(role, blocks):
        if msgs and msgs[-1]["role"] == role: msgs[-1]["content"].extend(blocks)
        else: msgs.append({"role": role, "content": list(blocks)})
    for t, m in turns:
        if t == "system": system.append({"type": "text", "text": m.get("text", "")})
        elif t in ("user", "synthetic"): push("user", [{"type": "text", "text": m.get("text", "") or "."}])
        elif t == "compaction": push("user", [{"type": "text", "text": "[summary] " + (m.get("summary") or "")}])
        elif t == "assistant":
            a, r = [], []
            for p in m.get("content", []):
                if p.get("type") == "text" and p.get("text"): a.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "tool":
                    a.append({"type": "tool_use", "id": p["id"], "name": p["name"], "input": (p.get("state") or {}).get("input") or {}})
                    r.append({"type": "tool_result", "tool_use_id": p["id"], "content": tool_out(p)})
            if a: push("assistant", a)
            if r: push("user", r)
    if system: system[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    if msgs: msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {"model": model, "max_tokens": 32000, "system": system, "messages": msgs, "stream": True,
            "tools": [{"name": n, "description": "Tool " + n, "input_schema": schema(n)} for n in tools_of(turns)],
            "thinking": {"type": "enabled", "budget_tokens": 10000}, "metadata": {"user_id": "user_42_session_7"}}

def responses(turns, model):
    items, instr, n = [], "", 0
    for t, m in turns:
        if t == "system": instr += m.get("text", "") + "\n"
        elif t in ("user", "synthetic"): items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": m.get("text", "")}]})
        elif t == "compaction": items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "[summary] " + (m.get("summary") or "")}]})
        elif t == "assistant":
            for p in m.get("content", []):
                n += 1
                if p.get("type") == "reasoning":
                    items.append({"type": "reasoning", "id": f"rs_{n:08x}", "summary": [{"type": "summary_text", "text": p.get("text", "")[:400]}], "encrypted_content": "gAAAA" + ("%08x" % n) * 40})
                elif p.get("type") == "text" and p.get("text"):
                    items.append({"type": "message", "role": "assistant", "id": f"msg_{n:08x}", "status": "completed", "content": [{"type": "output_text", "text": p["text"], "annotations": []}]})
                elif p.get("type") == "tool":
                    items.append({"type": "function_call", "id": f"fc_{n:08x}", "call_id": p["id"], "name": p["name"], "arguments": json.dumps((p.get("state") or {}).get("input") or {}, ensure_ascii=False), "status": "completed"})
                    items.append({"type": "function_call_output", "call_id": p["id"], "output": tool_out(p)})
    return {"model": model, "instructions": instr, "input": items, "stream": True, "store": False,
            "tools": [{"type": "function", "name": n_, "description": "Tool " + n_, "parameters": schema(n_), "strict": False} for n_ in tools_of(turns)],
            "tool_choice": "auto", "parallel_tool_calls": True, "reasoning": {"effort": "medium", "summary": "auto"},
            "include": ["reasoning.encrypted_content"], "prompt_cache_key": "ses-7", "text": {"verbosity": "medium"}}

# Each session is written in a different client serialisation style.
styles = [dict(ensure_ascii=False, separators=(",", ":")), dict(ensure_ascii=True, separators=(",", ":")),
          dict(ensure_ascii=False, separators=(", ", ": ")), dict(ensure_ascii=False, indent=2),
          dict(ensure_ascii=True, indent=1), dict(ensure_ascii=False, separators=(",", ":"), sort_keys=True)]
for i, sid in enumerate(sessions):
    turns = load(sid)
    for kind, build in (("chat", chat), ("messages", messages), ("responses", responses)):
        body = json.dumps(build(turns, "gpt-4o"), **styles[i % len(styles)])
        path = os.path.join(out, f"{kind}_{i}.json")
        open(path, "w", encoding="utf-8", newline="").write(body)
        print(f"{kind}_{i}.json  {len(body.encode('utf-8'))/1e6:6.2f} MB  turns={len(turns)}")
