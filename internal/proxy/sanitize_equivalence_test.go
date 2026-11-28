package proxy

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/mixaill76/auto_ai_router/internal/jsonscan"
)

// sanitizeJSONRequestBody rebuilds a changed body by copying its untouched
// members instead of decoding and re-encoding it. sanitizeJSONRequestBodyDecode
// is the previous implementation; every decision must match it, and the body
// must be the same JSON (not the same bytes: the old path also sorted keys).

// decodeForComparison decodes with number literals kept as written, so a
// rewrite that altered a number in any way is caught.
func decodeForComparison(doc []byte) (interface{}, bool) {
	dec := json.NewDecoder(bytes.NewReader(doc))
	dec.UseNumber()
	var v interface{}
	if err := dec.Decode(&v); err != nil {
		return nil, false
	}
	if dec.More() {
		return nil, false
	}
	return v, true
}

// sameJSONDocument reports whether a and b are the same JSON value, or the
// same bytes when they are not JSON at all.
func sameJSONDocument(a, b []byte) bool {
	if bytes.Equal(a, b) {
		return true
	}
	// Decoding replaces invalid UTF-8, so two bodies can decode alike while
	// only one of them is valid text on the wire: that must match too.
	if utf8.Valid(a) != utf8.Valid(b) {
		return false
	}
	av, aok := decodeForComparison(a)
	bv, bok := decodeForComparison(b)
	return aok && bok && reflect.DeepEqual(av, bv)
}

func assertSanitizeMatchesDecode(t testing.TB, body []byte, isMessagesAPI bool) {
	t.Helper()
	got, gotErr := sanitizeJSONRequestBody(body, isMessagesAPI)
	want, wantErr := sanitizeJSONRequestBodyDecode(body, isMessagesAPI)
	if (gotErr == nil) != (wantErr == nil) {
		t.Fatalf("error %v, decode path says %v\nbody %.300q", gotErr, wantErr, body)
	}
	gotBody, wantBody := got.Body, want.Body
	// The scan handed on with the body must be a scan of that very body, and
	// say what a fresh scan of it says.
	if got.scan != nil {
		fresh, ok := jsonscan.ScanTop(gotBody)
		if !ok || !got.scan.Describes(gotBody) {
			t.Fatalf("scan does not describe the returned body %.300q", gotBody)
		}
		var a, b []string
		got.scan.Range(func(name string, raw []byte) { a = append(a, name+"="+string(raw)) })
		fresh.Range(func(name string, raw []byte) { b = append(b, name+"="+string(raw)) })
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("scan layout differs from a fresh scan of %.300q", gotBody)
		}
	}
	got.Body, want.Body, got.scan, want.scan = nil, nil, nil, nil
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("result %+v, decode path says %+v\nbody %.300q", got, want, body)
	}
	if !sameJSONDocument(gotBody, wantBody) {
		t.Fatalf("body differs from the decode path\n got %.400q\nwant %.400q\n for %.300q", gotBody, wantBody, body)
	}
}

var sanitizeEquivalenceBodies = []string{
	``, `null`, `[]`, `{`, `{"model":"m"`, `{"model":"m"} x`, `{}`,
	`{"model":"m"}`, `{"model":7,"stream":true}`, `{"model":null,"service_tier":"priority"}`, `{"stream":true}`,
	`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
	`{"model":"m","messages":[],"stream":true}`,
	`{"model":"m","messages":[],"stream":"true"}`, `{"model":"m","messages":[],"stream":false}`, `{"model":"m","messages":[],"stream":null}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":true}}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":false}}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":false,"include_usage":true}}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":{ "include_usage" : true }}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":null}`, `{"model":"m","messages":[],"stream":true,"stream_options":"x"}`,
	`{"model":"m","messages":[],"stream":true,"stream_options":[1]}`, `{"model":"m","messages":[],"stream":true,"stream_options":{}}`,
	`{"model":"m","input":"hi","stream":true}`, `{"model":"m","input":[],"messages":[],"stream":true}`,
	`{"model":"m","input":"hi","stream":true,"background":true}`, `{"model":"m","input":"hi","background":false}`,
	`{"model":"m","prompt":"p","background":"transparent"}`,
	`{"model":"m","messages":[],"service_tier":"priority"}`, `{"model":"m","messages":[],"service_tier":""}`, `{"model":"m","messages":[],"service_tier":5}`,
	`{"model":"m","messages":[],"deferred":true,"search_parameters":{"mode":"on"}}`,
	`{"model":"m","messages":[],"extra_body":{"service_tier":"flex","keep":1}}`,
	`{"model":"m","messages":[],"extra_body":{"service_tier":"flex"},"service_tier":"priority","stream":true}`,
	`{"model":"m","messages":[],"extra_body":{"keep":1}}`, `{"model":"m","messages":[],"extra_body":"x"}`, `{"model":"m","messages":[],"extra_body":null}`,
	`{"model":"m","messages":[],"extra_body":{"background":true,"deferred":false}}`,
	`{"model":"m","messages":[],"extra_body":{"litellm_session_id":"e1","chat_id":"c1","session_id":"e2"}}`,
	`{"model":"m","messages":[],"extra_body":{"chat_id":"c1"}}`, `{"model":"m","messages":[],"extra_body":{"session_id":""}}`,
	`{"model":"m","messages":[],"litellm_session_id":"l1","session_id":"s1","prompt_cache_key":"p1"}`,
	`{"model":"m","messages":[],"session_id":"s1"}`, `{"model":"m","messages":[],"session_id":7}`, `{"model":"m","messages":[],"prompt_cache_key":"p1"}`,
	`{"model":"m","messages":[],"metadata":{"session_id":"m1"}}`, `{"model":"m","messages":[],"metadata":"x"}`,
	`{"model":"m","messages":[],"safety_identifier":"sid","user":"u"}`, `{"model":"m","messages":[],"user":"u"}`, `{"model":"m","messages":[],"user":{"id":1}}`,
	`{"model":"m","messages":[],"inference_geo":"eu","trace":{"a":1}}`,
	`{"model":"m","messages":[],"session_id":"a","session_id":"b","stream":true}`,
	`{"model":"first","model":"second","messages":[]}`, `{"model":"first","model":"second","messages":[],"stream":true}`,
	`{"model":"m","messages":[{"content":"a"}],"messages":[{"content":"b"}],"trace":1}`,
	`{"mo\u0064el":"m","messages":[],"str\u0065am":true}`, `{"model":"\u043c\u043e\u0434\u0435\u043b\u044c","messages":[],"session_id":"s"}`,
	"{\n  \"model\": \"m\",\n  \"messages\": [ { \"role\": \"user\", \"content\": \"<b>a & b</b> \\u2028\" } ],\n  \"stream\": true,\n  \"seed\": 12345678901234567890,\n  \"temperature\": 1.0\n}",
	`{"model":"m","messages":[],"stream":true,"big":1e999}`,
	"{\"model\":\"m\",\"messages\":[{\"content\":\"\xff\xfe\"}],\"stream\":true}",
	`{"model":"m","messages":[{"role":"user","content":"service_tier session_id stream_options in text"}],"tools":[{"function":{"parameters":{"properties":{"service_tier":{},"session_id":{}}}}}],"stream":true}`,
}

func TestSanitizeJSONRequestBodyMatchesDecodePath(t *testing.T) {
	for _, body := range sanitizeEquivalenceBodies {
		for _, isMessages := range []bool{false, true} {
			assertSanitizeMatchesDecode(t, []byte(body), isMessages)
		}
	}
}

func TestSanitizeJSONRequestBodyKeepsClientBytesWhenNothingChanges(t *testing.T) {
	// A streaming chat request that already asks for usage needs no rewrite:
	// the body that goes on is the client's own slice.
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	got, err := sanitizeJSONRequestBody(body, false)
	if err != nil || !got.Changed || !got.Streaming {
		t.Fatalf("got %+v, err %v", got, err)
	}
	if &got.Body[0] != &body[0] || len(got.Body) != len(body) {
		t.Fatal("expected the request body to be passed on without a copy")
	}
	// One that does not: only the small member is replaced, the rest is copied as sent.
	body = []byte(`{"model":"m","messages":[ {"role":"user","content":"a  b"} ],"stream":true}`)
	got, _ = sanitizeJSONRequestBody(body, false)
	want := `{"model":"m","messages":[ {"role":"user","content":"a  b"} ],"stream":true,"stream_options":{"include_usage":true}}`
	if string(got.Body) != want {
		t.Fatalf("got  %s\nwant %s", got.Body, want)
	}
}

func TestSanitizeJSONRequestBodyMatchesDecodePathOnCorpus(t *testing.T) {
	for _, base := range loadCorpus(t) {
		for _, c := range corpusVariants(t, base) {
			assertSanitizeMatchesDecode(t, c.body, c.kind == "messages")
		}
	}
}

func FuzzSanitizeJSONRequestBody(f *testing.F) {
	for _, body := range sanitizeEquivalenceBodies {
		f.Add([]byte(body), false)
		f.Add([]byte(body), true)
	}
	f.Fuzz(func(t *testing.T, body []byte, isMessages bool) {
		if strings.Contains(string(body), "\x00") {
			t.Skip()
		}
		assertSanitizeMatchesDecode(t, body, isMessages)
	})
}
