import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

const BASE = (__ENV.BASE_URL || 'https://example.ru');

const API_KEY     = __ENV.API_KEY || 'placeholder';
const AUTH_HEADER = __ENV.AUTH_HEADER || 'Authorization';
const AUTH_VALUE  = __ENV.AUTH_VALUE || (API_KEY ? `Bearer ${API_KEY}` : '');
const MODEL  = __ENV.MODEL || 'o4-mini';

const T_LOW  = Number(__ENV.TARGET_LOW  || 1000);
const T_HIGH = Number(__ENV.TARGET_HIGH || 3000);
const RAMP   = __ENV.RAMP || '2m';
const HOLD   = __ENV.HOLD || '2m';

const W_STREAM = Number(__ENV.W_STREAM || 30);
const W_CHAT   = Number(__ENV.W_CHAT   || 30);
const W_RESP   = Number(__ENV.W_RESP   || 30);
const W_EMBED  = Number(__ENV.W_EMBED  || 10);
const W_TOTAL  = W_STREAM + W_CHAT + W_RESP + W_EMBED;

const TRICKLE = __ENV.MOCK_TRICKLE_MS || '';

// --- метрики ---------------------------------------------------------------
const ttft        = new Trend('llm_ttft_ms', true);
const streamDur   = new Trend('llm_stream_duration_ms', true);
const perStreamTps = new Trend('llm_stream_tokens_per_sec');
const outTokens   = new Counter('llm_output_tokens');
const inTokens    = new Counter('llm_input_tokens');
const badBody     = new Rate('llm_bad_body');

// --- сценарий нагрузки ------------------------------------------------------
export const options = {
  discardResponseBodies: false,
  scenarios: {
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      gracefulRampDown: '30s',
      gracefulStop: '30s',
      stages: [
        { duration: RAMP, target: T_LOW },  
        { duration: HOLD, target: T_LOW },
        { duration: RAMP, target: T_HIGH },
        { duration: HOLD, target: T_HIGH },
        { duration: RAMP, target: 0 },
      ],
    },
  },
  thresholds: {
    http_req_failed:  ['rate<0.01'],
    checks:           ['rate>0.99'],
    llm_bad_body:     ['rate<0.01'],
    llm_ttft_ms:      ['p(95)<1500', 'p(99)<3000'],
    'llm_ttft_ms{endpoint:chat}':          ['p(95)<1500'],
    'llm_ttft_ms{endpoint:chat_nostream}': ['p(95)<3000'],
    'llm_ttft_ms{endpoint:embed}':         ['p(95)<500'],
  },
};

function headers(stream) {
  const h = { 'Content-Type': 'application/json' };
  if (stream) h['Accept'] = 'text/event-stream';
  if (AUTH_VALUE !== '') h[AUTH_HEADER] = AUTH_VALUE;
  if (TRICKLE !== '') h['X-Vidai-Chaos-Trickle'] = String(TRICKLE);
  return h;
}

function grab(re, body) {
  if (typeof body !== 'string') return 0;
  const m = re.exec(body);
  return m ? parseInt(m[1], 10) : 0;
}

function recordStream(res, ep, outRe, inRe) {
  const ok = check(res, {
    'status 200':      (r) => r.status === 200,
    'is event-stream': (r) => (r.headers['Content-Type'] || '').indexOf('text/event-stream') !== -1,
    'has [DONE]':      (r) => typeof r.body === 'string' && r.body.indexOf('[DONE]') !== -1,
  }, { endpoint: ep });

  ttft.add(res.timings.waiting, { endpoint: ep });
  streamDur.add(res.timings.duration, { endpoint: ep });

  const out = grab(outRe, res.body);
  const inp = grab(inRe, res.body);
  if (out > 0) {
    outTokens.add(out, { endpoint: ep });
    const secs = res.timings.duration / 1000;
    if (secs > 0) perStreamTps.add(out / secs, { endpoint: ep });
  }
  if (inp > 0) inTokens.add(inp, { endpoint: ep });

  badBody.add(!(ok && out > 0), { endpoint: ep });
}

// --- запросы ----------------------------------------------------------------
// стриминговый chat
function doChatStream() {
  const body = JSON.stringify({
    model: MODEL,
    stream: true,
    stream_options: { include_usage: true },
    messages: [{ role: 'user', content: 'load test' }],
  });
  const res = http.post(`${BASE}/v1/chat/completions`, body,
    { headers: headers(true), tags: { endpoint: 'chat' } });
  recordStream(res, 'chat', /"completion_tokens":\s*(\d+)/, /"prompt_tokens":\s*(\d+)/);
}

// обычный chat
function doChatNoStream() {
  const body = JSON.stringify({
    model: MODEL,
    stream: false,
    messages: [{ role: 'user', content: 'load test' }],
  });
  const res = http.post(`${BASE}/v1/chat/completions`, body,
    { headers: headers(false), tags: { endpoint: 'chat_nostream' } });

  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'is json':    (r) => (r.headers['Content-Type'] || '').indexOf('application/json') !== -1,
    'has usage':  (r) => typeof r.body === 'string' && r.body.indexOf('completion_tokens') !== -1,
  }, { endpoint: 'chat_nostream' });

  ttft.add(res.timings.waiting, { endpoint: 'chat_nostream' });
  streamDur.add(res.timings.duration, { endpoint: 'chat_nostream' });

  const out = grab(/"completion_tokens":\s*(\d+)/, res.body);
  const inp = grab(/"prompt_tokens":\s*(\d+)/, res.body);
  if (out > 0) {
    outTokens.add(out, { endpoint: 'chat_nostream' });
    const secs = res.timings.duration / 1000;
    if (secs > 0) perStreamTps.add(out / secs, { endpoint: 'chat_nostream' });
  }
  if (inp > 0) inTokens.add(inp, { endpoint: 'chat_nostream' });

  badBody.add(!(ok && out > 0), { endpoint: 'chat_nostream' });
}

// responses
function doResponses() {
  const body = JSON.stringify({
    model: MODEL,
    stream: true,
    input: 'load test',
  });
  const res = http.post(`${BASE}/v1/responses`, body,
    { headers: headers(true), tags: { endpoint: 'responses' } });
  recordStream(res, 'responses', /"output_tokens":\s*(\d+)/, /"input_tokens":\s*(\d+)/);
}

// embeddings
function doEmbed() {
  const body = JSON.stringify({ model: 'text-embedding-3-small', input: 'load test' });
  const res = http.post(`${BASE}/v1/embeddings`, body,
    { headers: headers(false), tags: { endpoint: 'embed' } });

  const ok = check(res, {
    'status 200':     (r) => r.status === 200,
    'has embedding':  (r) => typeof r.body === 'string' && r.body.indexOf('"embedding"') !== -1,
  }, { endpoint: 'embed' });

  ttft.add(res.timings.waiting, { endpoint: 'embed' });
  streamDur.add(res.timings.duration, { endpoint: 'embed' });
  const inp = grab(/"prompt_tokens":\s*(\d+)/, res.body);
  if (inp > 0) inTokens.add(inp, { endpoint: 'embed' });
  badBody.add(!(ok && inp > 0), { endpoint: 'embed' });
}

// --- жизненный цикл ---------------------------------------------------------
export function setup() {
  if (AUTH_VALUE === '') {
    console.log('[k6] WARNING: авторизация не задана (нет API_KEY / AUTH_VALUE) — шлюз, скорее всего, вернёт 401');
  }
  const res = http.get(`${BASE}/health`, { headers: headers(false) });
  if (res.status === 401 || res.status === 403) {
    throw new Error(`${BASE}/health вернул ${res.status} — проверьте API_KEY/заголовок авторизации`);
  }
  if (res.status !== 200) {
    throw new Error(`шлюз не отвечает на ${BASE}/health (status ${res.status})`);
  }
  console.log(`[k6] ${BASE} | auth=${AUTH_VALUE ? AUTH_HEADER : 'нет'} | mix stream/chat/resp/embed=${W_STREAM}/${W_CHAT}/${W_RESP}/${W_EMBED} | VU ${T_LOW}->${T_HIGH}`);
  if (TRICKLE !== '') {
    console.log(`[k6] note: X-Vidai-Chaos-Trickle=${TRICKLE} отправляется (актуально только для static-sse мока)`);
  }
}

export default function () {
  const r = Math.random() * W_TOTAL;
  if      (r < W_STREAM)                        doChatStream();
  else if (r < W_STREAM + W_CHAT)               doChatNoStream();
  else if (r < W_STREAM + W_CHAT + W_RESP)      doResponses();
  else                                          doEmbed();
}