package vertex

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/textproto"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

// gemini-nano-banana-2.1 carries none of the substrings ("image", "gemini-3",
// "flash") the rest of this package keys capabilities on: everything below comes
// from its geminiModelProfile.
const nanoBanana21 = "gemini-nano-banana-2.1"

// unsupportedSamplingKeys are the generationConfig fields the model rejects.
var unsupportedSamplingKeys = []string{"temperature", "topP", "topK", "seed", "responseLogprobs", "logprobs"}

func TestLookupGeminiModelProfile(t *testing.T) {
	for _, model := range []string{nanoBanana21, "GEMINI-NANO-BANANA-2.1", "publishers/google/models/gemini-nano-banana-2.1",
		"gemini-nano-banana-2.1-preview-10-2026", "gemini-nano-banana-2.1-001", "gemini-nano-banana-2.1@001"} {
		assert.NotNil(t, lookupGeminiModelProfile(model), model)
	}
	for _, model := range []string{"", "gemini-nano-banana-2.1-lite", "gemini-nano-banana-2.10", "gemini-3.1-flash-image"} {
		assert.Nil(t, lookupGeminiModelProfile(model), model)
	}
	// Detection comes from the ID, not from the "image" / "gemini-3" substrings.
	assert.True(t, isImageModel(nanoBanana21))
	assert.True(t, isThinkingCapableModel(nanoBanana21))
	assert.Equal(t, []string{"1K", "2K", "4K"}, geminiImageProfileForModel(nanoBanana21).resolutionValues())
}

func TestNanoBanana21ImageSize(t *testing.T) {
	// Sizes off the official grid (see below). 512px, a 3.1 Flash Image resolution,
	// is never picked, exactly or as the nearest size.
	tests := []struct{ size, aspectRatio, imageSize string }{
		{"512x512", "1:1", "1K"},
		{"128x128", "1:1", "1K"},
		{"1792x1024", "16:9", "1K"},
		{"1:8", "1:8", "1K"},
		{"20000x20000", "1:1", "4K"},
	}
	for _, tt := range tests {
		t.Run(tt.size, func(t *testing.T) {
			config, err := mapGeminiImageSize(nanoBanana21, tt.size)
			require.NoError(t, err)
			require.NotNil(t, config)
			assert.Equal(t, tt.aspectRatio, config.aspectRatio)
			assert.Equal(t, tt.imageSize, config.imageSize)
		})
	}
}

// TestNanoBanana21ImageSizeOfficialGrid is the model's resolution table from the
// Gemini API image generation docs: every listed size maps back to its ratio and
// resolution.
func TestNanoBanana21ImageSizeOfficialGrid(t *testing.T) {
	grid := map[string][3]string{
		"1:1":  {"1024x1024", "2048x2048", "4096x4096"},
		"1:4":  {"512x2048", "1024x4096", "2048x8192"},
		"1:8":  {"384x3072", "768x6144", "1536x12288"},
		"2:3":  {"848x1264", "1696x2528", "3392x5056"},
		"3:2":  {"1264x848", "2528x1696", "5056x3392"},
		"3:4":  {"896x1200", "1792x2400", "3584x4800"},
		"4:1":  {"2048x512", "4096x1024", "8192x2048"},
		"4:3":  {"1200x896", "2400x1792", "4800x3584"},
		"4:5":  {"928x1152", "1856x2304", "3712x4608"},
		"5:4":  {"1152x928", "2304x1856", "4608x3712"},
		"8:1":  {"3072x384", "6144x768", "12288x1536"},
		"9:16": {"768x1376", "1536x2752", "3072x5504"},
		"16:9": {"1376x768", "2752x1536", "5504x3072"},
		"21:9": {"1584x672", "3168x1344", "6336x2688"},
	}
	for aspectRatio, sizes := range grid {
		for i, imageSize := range []string{"1K", "2K", "4K"} {
			t.Run(aspectRatio+"/"+imageSize, func(t *testing.T) {
				config, err := mapGeminiImageSize(nanoBanana21, sizes[i])
				require.NoError(t, err)
				require.NotNil(t, config)
				assert.Equal(t, aspectRatio, config.aspectRatio)
				assert.Equal(t, imageSize, config.imageSize)
			})
		}
	}
}

func TestNanoBanana21ExplicitImageConfig(t *testing.T) {
	t.Run("accepted and normalized", func(t *testing.T) {
		req := nanoChat(t, `"image_config":{"aspect_ratio":"8:1","image_size":"2k"}`)
		imageConfig := mapAt(t, generationConfigOf(t, req), "imageConfig")
		assert.Equal(t, "8:1", imageConfig["aspectRatio"])
		assert.Equal(t, "2K", imageConfig["imageSize"])
	})

	rejected := []struct {
		name   string
		params string
		param  string
	}{
		{name: "512 is not a 2.1 resolution", params: `"image_size":"512"`, param: "image_size"},
		{name: "unknown resolution", params: `"imageSize":"8K"`, param: "image_size"},
		// Listed by Vertex only; not a shared capability until both routes accept it.
		{name: "9:21 is not in the shared grid", params: `"aspect_ratio":"9:21"`, param: "aspect_ratio"},
		{name: "unknown ratio", params: `"extra_body":{"generation_config":{"image_config":{"aspectRatio":"2:1"}}}`, param: "aspect_ratio"},
	}
	for _, tt := range rejected {
		t.Run(tt.name, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Draw"}],%s}`, nanoBanana21, tt.params)
			_, err := OpenAIToVertex([]byte(body), false, false, nanoBanana21, "application/json")
			testhelpers.RequireValidationError(t, err, tt.param, "invalid_value")
		})
	}

	t.Run("images endpoint rejects 512 before the upstream call", func(t *testing.T) {
		body := fmt.Sprintf(`{"model":%q,"prompt":"A cat","image_size":"512"}`, nanoBanana21)
		_, err := OpenAIToVertex([]byte(body), true, false, nanoBanana21, "application/json")
		testhelpers.RequireValidationError(t, err, "image_size", "invalid_value")
	})

	t.Run("models without a profile pass the grid unchecked, image_size upper-cased", func(t *testing.T) {
		body := `{"model":"gemini-3.1-flash-image","messages":[{"role":"user","content":"Draw"}],"aspect_ratio":"9:21","image_size":"2k"}`
		out, err := OpenAIToVertex([]byte(body), false, false, "gemini-3.1-flash-image", "application/json")
		require.NoError(t, err)
		var req map[string]interface{}
		require.NoError(t, json.Unmarshal(out, &req))
		imageConfig := mapAt(t, generationConfigOf(t, req), "imageConfig")
		assert.Equal(t, "9:21", imageConfig["aspectRatio"])
		assert.Equal(t, "2K", imageConfig["imageSize"])
	})
}

func TestNanoBanana21Thinking(t *testing.T) {
	tests := []struct {
		name            string
		params          string
		level           genai.ThinkingLevel
		includeThoughts bool
	}{
		{name: "default without thinking params", params: `"max_tokens":1000`, level: genai.ThinkingLevelMedium},
		{name: "reasoning_effort low becomes minimal", params: `"reasoning_effort":"low"`, level: genai.ThinkingLevelMinimal},
		{name: "reasoning_effort xhigh becomes high", params: `"reasoning_effort":"xhigh"`, level: genai.ThinkingLevelHigh},
		{name: "reasoning_effort none is the lowest level", params: `"reasoning_effort":"none"`, level: genai.ThinkingLevelMinimal},
		{name: "unknown reasoning_effort keeps the default", params: `"reasoning_effort":"auto"`, level: genai.ThinkingLevelMedium},
		{name: "thinking_level LOW in upper case", params: `"thinking_level":"LOW"`, level: genai.ThinkingLevelMinimal},
		{name: "thinkingLevel camelCase", params: `"thinkingLevel":"high"`, level: genai.ThinkingLevelHigh},
		{
			name:   "top-level thinkingConfig camelCase",
			params: `"thinkingConfig":{"thinkingLevel":"minimal","includeThoughts":true}`,
			level:  genai.ThinkingLevelMinimal, includeThoughts: true,
		},
		{
			name:   "extra_body generation_config thinking_config",
			params: `"extra_body":{"generation_config":{"thinking_config":{"thinking_level":"high"}}}`,
			level:  genai.ThinkingLevelHigh,
		},
		// A budget is never sent: it maps to the level of the same depth.
		{name: "zero thinking_budget is the lowest level", params: `"thinking_budget":0`, level: genai.ThinkingLevelMinimal},
		{name: "dynamic thinking_budget keeps the default", params: `"thinking_budget":-1`, level: genai.ThinkingLevelMedium},
		{name: "thinking_config budget", params: `"thinking_config":{"thinkingBudget":20000}`, level: genai.ThinkingLevelHigh},
		{name: "thinking_level wins over thinking_budget", params: `"thinking_level":"minimal","thinking_budget":20000`, level: genai.ThinkingLevelMinimal},
		{name: "extra_body reasoning_effort", params: `"extra_body":{"reasoning_effort":"low"}`, level: genai.ThinkingLevelMinimal},
		{name: "anthropic medium budget", params: `"thinking":{"type":"enabled","budget_tokens":8000}`, level: genai.ThinkingLevelMedium},
		{name: "anthropic disabled", params: `"thinking":{"type":"disabled"}`, level: genai.ThinkingLevelMinimal},
		{name: "anthropic adaptive keeps the default", params: `"thinking":{"type":"adaptive"}`, level: genai.ThinkingLevelMedium},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := nanoChat(t, tt.params)
			thinking := mapAt(t, generationConfigOf(t, req), "thinkingConfig")
			assert.Equal(t, string(tt.level), thinking["thinkingLevel"])
			assert.NotContains(t, thinking, "thinkingBudget")
			if tt.includeThoughts {
				assert.Equal(t, true, thinking["includeThoughts"])
			} else {
				assert.NotContains(t, thinking, "includeThoughts")
			}
		})
	}

	t.Run("no generation params leaves the provider default", func(t *testing.T) {
		req := nanoChat(t, `"stream":false`)
		assert.NotContains(t, req, "generationConfig")
	})
}

func TestNanoBanana21DropsUnsupportedSamplingParams(t *testing.T) {
	generationConfig := generationConfigOf(t, nanoChat(t, `"temperature":0.7,"top_p":0.9,"seed":42,"logprobs":true,`+
		`"top_logprobs":3,"extra_body":{"generation_config":{"temperature":0.5,"top_k":40,"seed":7}}`))
	for _, key := range unsupportedSamplingKeys {
		assert.NotContains(t, generationConfig, key)
	}
	assert.Equal(t, "MEDIUM", mapAt(t, generationConfig, "thinkingConfig")["thinkingLevel"], "the rest is unaffected")

	t.Run("images generations", func(t *testing.T) {
		req := nanoImageGeneration(t, fmt.Sprintf(`{"model":%q,"prompt":"A cat","temperature":0.7,"top_p":0.9,"seed":42}`, nanoBanana21))
		generationConfig := generationConfigOf(t, req)
		for _, key := range unsupportedSamplingKeys {
			assert.NotContains(t, generationConfig, key)
		}
	})

	t.Run("multipart images edits", func(t *testing.T) {
		body, contentType := nanoMultipartEdit(t, map[string]string{"temperature": "0.7", "top_p": "0.9", "seed": "42"}, 1, false)
		req := nanoImageEdit(t, body, contentType)
		generationConfig := generationConfigOf(t, req)
		for _, key := range unsupportedSamplingKeys {
			assert.NotContains(t, generationConfig, key)
		}
	})

	t.Run("other Gemini image models keep them", func(t *testing.T) {
		temperature := 0.7
		cfg := buildGenerationConfig(&openai.OpenAIRequest{Model: "gemini-3.1-flash-image", Temperature: &temperature}, "gemini-3.1-flash-image")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.Temperature)
		assert.InDelta(t, 0.7, *cfg.Temperature, 1e-6)
	})
}

func TestGoogleSearchTypes(t *testing.T) {
	tests := []struct {
		name  string
		tools []interface{}
		want  string
	}{
		{name: "no search_types is web search", tools: []interface{}{
			map[string]interface{}{"type": "web_search"},
		}, want: `[{"googleSearch":{}}]`},
		{name: "Gemini object form", tools: []interface{}{
			map[string]interface{}{"type": "google_search", "search_types": map[string]interface{}{"imageSearch": map[string]interface{}{}}},
		}, want: `[{"googleSearch":{"searchTypes":{"imageSearch":{}}}}]`},
		{name: "object form entry set to false or null is off", tools: []interface{}{
			map[string]interface{}{"type": "google_search", "search_types": map[string]interface{}{"webSearch": false, "imageSearch": true}},
			map[string]interface{}{"type": "google_search", "search_types": map[string]interface{}{"webSearch": nil, "imageSearch": map[string]interface{}{}}},
		}, want: `[{"googleSearch":{"searchTypes":{"imageSearch":{}}}}]`},
		{name: "search tools merge into one", tools: []interface{}{
			map[string]interface{}{"type": "web_search"},
			map[string]interface{}{"type": "google_search", "search_types": []interface{}{"image_search"}},
		}, want: `[{"googleSearch":{"searchTypes":{"webSearch":{},"imageSearch":{}}}}]`},
		{name: "image search twice stays image only", tools: []interface{}{
			map[string]interface{}{"type": "web_search", "search_types": []interface{}{"image_search"}},
			map[string]interface{}{"type": "web_search_preview", "search_types": []interface{}{"Image_Search"}},
		}, want: `[{"googleSearch":{"searchTypes":{"imageSearch":{}}}}]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertOpenAIToolsToVertex(tt.tools)
			require.NoError(t, err)
			tools, err := json.Marshal(result.Tools)
			require.NoError(t, err)
			assert.JSONEq(t, tt.want, string(tools))
		})
	}

	invalid := []struct {
		name        string
		searchTypes interface{}
		param       string
		code        string
	}{
		{name: "unknown type", searchTypes: []interface{}{"news_search"}, code: "invalid_value"},
		{name: "unknown object key", searchTypes: map[string]interface{}{"newsSearch": map[string]interface{}{}}, code: "invalid_value"},
		{name: "not a list", searchTypes: "image_search", code: "invalid_type"},
		{name: "non-string entry", searchTypes: []interface{}{1.0}, code: "invalid_type"},
		{name: "bare alias", searchTypes: []interface{}{"image"}, code: "invalid_value"},
		{name: "object entry neither object nor bool", searchTypes: map[string]interface{}{"imageSearch": "no"},
			param: "tools[1].search_types.imageSearch", code: "invalid_type"},
	}
	for _, tt := range invalid {
		t.Run("rejects "+tt.name, func(t *testing.T) {
			_, err := convertOpenAIToolsToVertex([]interface{}{
				map[string]interface{}{"type": "web_search"},
				map[string]interface{}{"type": "web_search", "search_types": tt.searchTypes},
			})
			testhelpers.RequireValidationError(t, err, cmp.Or(tt.param, "tools[1].search_types"), tt.code)
		})
	}
}

func TestCountWebSearchRequests_CountsImageSearchQueries(t *testing.T) {
	grounded := func(web, image []string) []*genai.Candidate {
		return []*genai.Candidate{{GroundingMetadata: &genai.GroundingMetadata{WebSearchQueries: web, ImageSearchQueries: image}}}
	}
	assert.Equal(t, 2, CountWebSearchRequests(grounded([]string{"a", "b", "a", " "}, nil)), "web only")
	assert.Equal(t, 2, CountWebSearchRequests(grounded(nil, []string{"butterfly", "flower", "butterfly"})), "image only")
	// The same text searched on the web and in images is two billed queries.
	assert.Equal(t, 3, CountWebSearchRequests(grounded([]string{"butterfly", "species"}, []string{"butterfly"})), "web and image")
	assert.Equal(t, 0, CountWebSearchRequests(grounded(nil, nil)), "no search")
}

func TestVertexToOpenAI_GoogleSearchGroundingBilling(t *testing.T) {
	response := func(grounding, extra, modelVersion string) []byte {
		return []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"` +
			grounding + extra + `}],"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":900,` +
			`"candidatesTokenCount":50,"totalTokenCount":1050},"modelVersion":"` + modelVersion + `"}`)
	}
	search := `,"groundingMetadata":{"webSearchQueries":["timareta butterfly"],"imageSearchQueries":["timareta butterfly","flower"]}`

	tests := []struct {
		name          string
		model         string
		body          []byte
		promptTokens  int
		totalTokens   int
		searchQueries int
	}{
		// Nano Banana 2.1 is priced without charging Google Search context as input.
		{name: "search grounding only", model: nanoBanana21, body: response(search, "", ""),
			promptTokens: 100, totalTokens: 150, searchQueries: 3},
		// The router may know the model by an alias; Gemini's modelVersion names it.
		{name: "search grounding only, model under an alias", model: "banana-alias", body: response(search, "", nanoBanana21+"-preview"),
			promptTokens: 100, totalTokens: 150, searchQueries: 3},
		// Any other model is billed for its search context as before.
		{name: "search grounding only, unprofiled model", model: "gemini-2.5-flash", body: response(search, "", "gemini-2.5-flash"),
			promptTokens: 1000, totalTokens: 1050, searchQueries: 3},
		// The answering model decides, not the name the router routed by: an alias
		// answered by a model that bills search context stays billed.
		{name: "search grounding only, answered by an unprofiled model", model: nanoBanana21,
			body: response(search, "", "gemini-3.1-flash-image"), promptTokens: 1000, totalTokens: 1050, searchQueries: 3},
		// url_context content is billed as input; the split is unknown, so all of it stays.
		{name: "search and url context", model: nanoBanana21,
			body:         response(search, `,"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com"}]}`, ""),
			promptTokens: 1000, totalTokens: 1050, searchQueries: 3},
		{name: "no search", model: nanoBanana21, body: response("", "", ""), promptTokens: 1000, totalTokens: 1050},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := VertexToOpenAI(tt.body, tt.model)
			require.NoError(t, err)
			var resp struct {
				Usage struct {
					PromptTokens  int `json:"prompt_tokens"`
					TotalTokens   int `json:"total_tokens"`
					ServerToolUse *struct {
						WebSearchRequests int `json:"web_search_requests"`
					} `json:"server_tool_use"`
				} `json:"usage"`
			}
			require.NoError(t, json.Unmarshal(out, &resp))
			assert.Equal(t, tt.promptTokens, resp.Usage.PromptTokens)
			assert.Equal(t, tt.totalTokens, resp.Usage.TotalTokens)
			if tt.searchQueries == 0 {
				assert.Nil(t, resp.Usage.ServerToolUse)
				return
			}
			require.NotNil(t, resp.Usage.ServerToolUse)
			assert.Equal(t, tt.searchQueries, resp.Usage.ServerToolUse.WebSearchRequests)
		})
	}
}

// TestTransformVertexStreamToOpenAI_GoogleSearchGroundingBilling streams usage on
// every chunk, as Gemini does, with the grounding metadata only on the last one.
func TestTransformVertexStreamToOpenAI_GoogleSearchGroundingBilling(t *testing.T) {
	usage := `"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":900,` +
		`"toolUsePromptTokensDetails":[{"modality":"IMAGE","tokenCount":600}],"candidatesTokenCount":50,"totalTokenCount":1050}`
	stream := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"Here"}]}}],` + usage + `}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":" it is"}]},"finishReason":"STOP",` +
		`"groundingMetadata":{"webSearchQueries":["q"],"imageSearchQueries":["q"]}}],` + usage + `}` + "\n\n"
	imageTokens := func(usage map[string]interface{}) float64 {
		details, _ := usage["prompt_tokens_details"].(map[string]interface{})
		tokens, _ := details["image_tokens"].(float64)
		return tokens
	}

	tests := []struct {
		name                         string
		model                        string
		firstImageTokens             float64
		lastPromptTokens, lastImages float64
	}{
		// Until the last chunk it is unknown whether the search context is billed,
		// so the early chunk leaves out its image tokens (a later zero could not
		// take them back); the last chunk drops the search context altogether.
		{name: "search context unbilled", model: nanoBanana21, lastPromptTokens: 100},
		// Billed as before: every chunk as Gemini sent it.
		{name: "search context billed", model: "gemini-2.5-flash", firstImageTokens: 600, lastPromptTokens: 1000, lastImages: 600},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			require.NoError(t, TransformVertexStreamToOpenAI(strings.NewReader(stream), tt.model, &out))

			var usages []map[string]interface{}
			finishReasons := 0
			for _, line := range strings.Split(out.String(), "\n") {
				payload, ok := strings.CutPrefix(line, "data: ")
				if !ok || payload == "[DONE]" {
					continue
				}
				var chunk map[string]interface{}
				require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
				if chunk["usage"] != nil {
					usages = append(usages, mapAt(t, chunk, "usage"))
				}
				for _, choice := range chunk["choices"].([]interface{}) {
					if choice.(map[string]interface{})["finish_reason"] != nil {
						finishReasons++
					}
				}
			}
			assert.Equal(t, 1, finishReasons, "only the chunk carrying Gemini's finish reason ends the choice")
			require.Len(t, usages, 2)
			assert.EqualValues(t, 1000, usages[0]["prompt_tokens"])
			assert.Equal(t, tt.firstImageTokens, imageTokens(usages[0]))
			assert.Equal(t, tt.lastPromptTokens, usages[1]["prompt_tokens"])
			assert.Equal(t, tt.lastImages, imageTokens(usages[1]))
			assert.EqualValues(t, 2, mapAt(t, usages[1], "server_tool_use")["web_search_requests"])
		})
	}
}

func TestNanoBanana21ReferenceImages(t *testing.T) {
	for _, count := range []int{14} {
		t.Run(fmt.Sprintf("JSON edit with %d images", count), func(t *testing.T) {
			images := make([]string, count)
			for i := range images {
				images[i] = testImageDataURL(i)
			}
			encoded, err := json.Marshal(images)
			require.NoError(t, err)
			body := fmt.Sprintf(`{"model":%q,"prompt":"Combine these","images":%s}`, nanoBanana21, encoded)
			req := nanoImageEdit(t, []byte(body), "application/json")
			assertReferenceImages(t, req, count)
		})
		t.Run(fmt.Sprintf("multipart edit with %d images", count), func(t *testing.T) {
			body, contentType := nanoMultipartEdit(t, nil, count, false)
			req := nanoImageEdit(t, body, contentType)
			assertReferenceImages(t, req, count)
		})
		t.Run(fmt.Sprintf("chat with %d images", count), func(t *testing.T) {
			req := nanoChatImages(t, count)
			assertReferenceImages(t, req, count)
		})
	}

	t.Run("15 images are rejected locally", func(t *testing.T) {
		_, err := OpenAIToVertex(nanoChatImagesBody(t, 15), false, false, nanoBanana21, "application/json")
		testhelpers.RequireValidationError(t, err, "image", "too_many_images")
	})

	t.Run("images from earlier turns count toward the limit", func(t *testing.T) {
		// Google's limit is per request: a multi-turn edit sends the whole history.
		turns := func(perTurn int) []byte {
			userTurn := func(offset int) map[string]interface{} {
				content := []interface{}{map[string]interface{}{"type": "text", "text": "Edit"}}
				for i := 0; i < perTurn; i++ {
					content = append(content, map[string]interface{}{"type": "image_url",
						"image_url": map[string]interface{}{"url": testImageDataURL(offset + i)}})
				}
				return map[string]interface{}{"role": "user", "content": content}
			}
			body, err := json.Marshal(map[string]interface{}{"model": nanoBanana21, "messages": []interface{}{
				userTurn(0), map[string]interface{}{"role": "assistant", "content": "Done"}, userTurn(perTurn),
			}})
			require.NoError(t, err)
			return body
		}
		_, err := OpenAIToVertex(turns(7), false, false, nanoBanana21, "application/json")
		require.NoError(t, err)
		_, err = OpenAIToVertex(turns(8), false, false, nanoBanana21, "application/json")
		testhelpers.RequireValidationError(t, err, "image", "too_many_images")
	})

	t.Run("a mask counts toward the limit", func(t *testing.T) {
		body, contentType := nanoMultipartEdit(t, nil, 14, true)
		_, err := OpenAIToVertex(body, false, true, nanoBanana21, contentType)
		testhelpers.RequireValidationError(t, err, "image", "too_many_images")
	})

	t.Run("models without a profile have no local cap", func(t *testing.T) {
		body, contentType := nanoMultipartEdit(t, nil, 15, false)
		_, err := OpenAIToVertex(body, false, true, "gemini-3.1-flash-image", contentType)
		require.NoError(t, err)
	})
}

// TestNanoBanana21ImageEndpointsNormalizeAlike sends the same Gemini options
// through images.generate (JSON), images.edit (JSON) and images.edit (multipart):
// all three must reach Google with the same generation config.
func TestNanoBanana21ImageEndpointsNormalizeAlike(t *testing.T) {
	tests := []struct {
		name          string
		options       map[string]string // multipart fields
		jsonOptions   string            // the same options in a JSON body
		aspectRatio   string
		imageSize     string
		thinkingLevel genai.ThinkingLevel
	}{
		{
			name:          "flat fields",
			options:       map[string]string{"aspect_ratio": "16:9", "image_size": "4k", "thinking_level": "high"},
			jsonOptions:   `"aspect_ratio":"16:9","image_size":"4k","thinking_level":"high"`,
			aspectRatio:   "16:9",
			imageSize:     "4K",
			thinkingLevel: genai.ThinkingLevelHigh,
		},
		{
			name:          "image_config object and camelCase thinking",
			options:       map[string]string{"image_config": `{"aspect_ratio":"21:9","image_size":"2K"}`, "thinkingLevel": "minimal"},
			jsonOptions:   `"image_config":{"aspect_ratio":"21:9","image_size":"2K"},"thinkingLevel":"minimal"`,
			aspectRatio:   "21:9",
			imageSize:     "2K",
			thinkingLevel: genai.ThinkingLevelMinimal,
		},
		{
			// The multipart encoding of image_config is accepted in a JSON body too.
			name:          "image_config as a JSON string",
			options:       map[string]string{"imageConfig": `{"aspectRatio":"4:5"}`},
			jsonOptions:   `"imageConfig":"{\"aspectRatio\":\"4:5\"}"`,
			aspectRatio:   "4:5",
			imageSize:     "1K",
			thinkingLevel: genai.ThinkingLevelMedium,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generation := nanoImageGeneration(t, fmt.Sprintf(`{"model":%q,"prompt":"Edit","size":"1024x1024",%s}`, nanoBanana21, tt.jsonOptions))
			jsonEdit := nanoImageEdit(t, []byte(fmt.Sprintf(`{"model":%q,"prompt":"Edit","size":"1024x1024","image":%q,%s}`,
				nanoBanana21, testImageDataURL(0), tt.jsonOptions)), "application/json")
			fields := map[string]string{"size": "1024x1024"}
			for key, value := range tt.options {
				fields[key] = value
			}
			multipartBody, contentType := nanoMultipartEdit(t, fields, 1, false)
			multipartEdit := nanoImageEdit(t, multipartBody, contentType)

			want := generationConfigOf(t, generation)
			imageConfig := mapAt(t, want, "imageConfig")
			assert.Equal(t, tt.aspectRatio, imageConfig["aspectRatio"])
			assert.Equal(t, tt.imageSize, imageConfig["imageSize"])
			assert.Equal(t, string(tt.thinkingLevel), mapAt(t, want, "thinkingConfig")["thinkingLevel"])
			assert.Equal(t, []interface{}{"IMAGE"}, want["responseModalities"])

			assert.Equal(t, want, generationConfigOf(t, jsonEdit), "JSON edit")
			assert.Equal(t, want, generationConfigOf(t, multipartEdit), "multipart edit")
		})
	}
}

// TestImageEndpointsThinkingOnlyForProfiledModels pins that the images endpoints'
// thinking fields reach only a model whose profile pins its thinking levels: other
// image models take no ThinkingConfig, as on the chat route.
func TestImageEndpointsThinkingOnlyForProfiledModels(t *testing.T) {
	for _, model := range []string{"gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview", "gemini-2.5-flash-image"} {
		for _, option := range []string{`"thinking_level":"medium"`, `"reasoning_effort":"high"`} {
			t.Run(model+" "+option, func(t *testing.T) {
				body := fmt.Sprintf(`{"model":%q,"prompt":"A cat",%s}`, model, option)
				out, err := OpenAIToVertex([]byte(body), true, false, model, "application/json")
				require.NoError(t, err)
				var req map[string]interface{}
				require.NoError(t, json.Unmarshal(out, &req))
				assert.NotContains(t, generationConfigOf(t, req), "thinkingConfig")
			})
		}
	}

	t.Run("malformed image_config is rejected", func(t *testing.T) {
		body := fmt.Sprintf(`{"model":%q,"prompt":"A cat","image_config":"{not json"}`, nanoBanana21)
		_, err := OpenAIToVertex([]byte(body), true, false, nanoBanana21, "application/json")
		testhelpers.RequireValidationError(t, err, "image_config", "invalid_json")
	})
}

// TestImageEndpointsExtrasOfWrongType: a JSON images request used to pass the Gemini
// extras unread, so a value of the wrong type must not start failing for a model
// without a profile; a profiled model, whose parameters are documented, rejects it.
func TestImageEndpointsExtrasOfWrongType(t *testing.T) {
	extras := []struct{ json, param, code string }{
		{json: `"image_size":2048`, param: "image_size", code: "invalid_type"},
		{json: `"aspectRatio":{"w":16,"h":9}`, param: "aspectRatio", code: "invalid_type"},
		{json: `"reasoning_effort":{"effort":"high"}`, param: "reasoning_effort", code: "invalid_type"},
		{json: `"thinkingLevel":3`, param: "thinkingLevel", code: "invalid_type"},
		{json: `"image_config":"{not json"`, param: "image_config", code: "invalid_json"},
		{json: `"imageConfig":[1,2]`, param: "imageConfig", code: "invalid_json"},
	}
	for _, tt := range extras {
		t.Run("unprofiled model ignores "+tt.json, func(t *testing.T) {
			const model = "gemini-3.1-flash-image-preview"
			for _, edit := range []bool{false, true} {
				body := fmt.Sprintf(`{"model":%q,"prompt":"A cat","size":"1024x1024","image":%q,%s}`, model, testImageDataURL(0), tt.json)
				out, err := OpenAIToVertex([]byte(body), !edit, edit, model, "application/json")
				require.NoError(t, err)
				imageConfig := mapAt(t, generationConfigOf(t, decodeVertexRequest(t, out)), "imageConfig")
				assert.Equal(t, map[string]interface{}{"aspectRatio": "1:1", "imageSize": "1K"}, imageConfig, "size still applies")
			}
		})
		t.Run("profiled model rejects "+tt.json, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"prompt":"A cat",%s}`, nanoBanana21, tt.json)
			_, err := OpenAIToVertex([]byte(body), true, false, nanoBanana21, "application/json")
			testhelpers.RequireValidationError(t, err, tt.param, tt.code)
		})
	}

	t.Run("a wrongly typed extra does not hide a valid one", func(t *testing.T) {
		const model = "gemini-3.1-flash-image-preview"
		body := fmt.Sprintf(`{"model":%q,"prompt":"A cat","image_size":2048,"aspect_ratio":"16:9"}`, model)
		out, err := OpenAIToVertex([]byte(body), true, false, model, "application/json")
		require.NoError(t, err)
		imageConfig := mapAt(t, generationConfigOf(t, decodeVertexRequest(t, out)), "imageConfig")
		assert.Equal(t, map[string]interface{}{"aspectRatio": "16:9"}, imageConfig)
	})
}

func TestConvertVertexUsageToImageUsage_Thinking(t *testing.T) {
	got := convertVertexUsageToImageUsage(&genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount: 20, CandidatesTokenCount: 1120, ThoughtsTokenCount: 300,
		CandidatesTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityImage, TokenCount: 1120}},
	}, 1)
	assert.Equal(t, 1420, got.OutputTokens, "thinking is output, apart from image tokens")
	require.NotNil(t, got.OutputTokensDetails)
	assert.Equal(t, 1120, got.OutputTokensDetails.ImageTokens)
	assert.Equal(t, 300, got.OutputTokensDetails.ReasoningTokens)
}

func TestVertexChatResponseToOpenAIImage_SkipsThoughtImages(t *testing.T) {
	draft := base64.StdEncoding.EncodeToString([]byte("draft"))
	final := base64.StdEncoding.EncodeToString([]byte("final"))
	body := `{"candidates":[{"content":{"role":"model","parts":[` +
		`{"thought":true,"inlineData":{"mimeType":"image/png","data":"` + draft + `"}},` +
		`{"inlineData":{"mimeType":"image/png","data":"` + final + `"}}]},"finishReason":"STOP"}]}`
	out, err := VertexChatResponseToOpenAIImageWithModel([]byte(body), nanoBanana21)
	require.NoError(t, err)
	var resp struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(out, &resp))
	require.Len(t, resp.Data, 1)
	assert.Equal(t, final, resp.Data[0].B64JSON)
}

// --- helpers ---

func nanoChat(t *testing.T, params string) map[string]interface{} {
	t.Helper()
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"Draw a banana"}],%s}`, nanoBanana21, params)
	out, err := OpenAIToVertex([]byte(body), false, false, nanoBanana21, "application/json")
	require.NoError(t, err)
	return decodeVertexRequest(t, out)
}

func nanoChatImagesBody(t *testing.T, count int) []byte {
	t.Helper()
	blocks := []interface{}{map[string]interface{}{"type": "text", "text": "Combine these"}}
	for i := 0; i < count; i++ {
		blocks = append(blocks, map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]interface{}{"url": testImageDataURL(i)},
		})
	}
	body, err := json.Marshal(map[string]interface{}{
		"model":    nanoBanana21,
		"messages": []interface{}{map[string]interface{}{"role": "user", "content": blocks}},
	})
	require.NoError(t, err)
	return body
}

func nanoChatImages(t *testing.T, count int) map[string]interface{} {
	t.Helper()
	out, err := OpenAIToVertex(nanoChatImagesBody(t, count), false, false, nanoBanana21, "application/json")
	require.NoError(t, err)
	return decodeVertexRequest(t, out)
}

func nanoImageGeneration(t *testing.T, body string) map[string]interface{} {
	t.Helper()
	out, err := OpenAIToVertex([]byte(body), true, false, nanoBanana21, "application/json")
	require.NoError(t, err)
	return decodeVertexRequest(t, out)
}

func nanoImageEdit(t *testing.T, body []byte, contentType string) map[string]interface{} {
	t.Helper()
	out, err := OpenAIToVertex(body, false, true, nanoBanana21, contentType)
	require.NoError(t, err)
	return decodeVertexRequest(t, out)
}

// nanoMultipartEdit builds a multipart images.edit request with count images
// (cycling PNG / JPEG / WebP so order and MIME type can be checked) and an
// optional mask.
func nanoMultipartEdit(t *testing.T, fields map[string]string, count int, mask bool) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	require.NoError(t, writer.WriteField("model", nanoBanana21))
	require.NoError(t, writer.WriteField("prompt", "Combine these"))
	for name, value := range fields {
		require.NoError(t, writer.WriteField(name, value))
	}
	writeImage := func(name, filename, mimeType string, data []byte) {
		part, err := writer.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": []string{fmt.Sprintf(`form-data; name=%q; filename=%q`, name, filename)},
			"Content-Type":        []string{mimeType},
		})
		require.NoError(t, err)
		_, err = part.Write(data)
		require.NoError(t, err)
	}
	for i := 0; i < count; i++ {
		mimeType, data := testImage(i)
		writeImage("image[]", fmt.Sprintf("ref-%d", i), mimeType, data)
	}
	if mask {
		writeImage("mask", "mask.png", "image/png", []byte("mask"))
	}
	require.NoError(t, writer.Close())
	return buf.Bytes(), writer.FormDataContentType()
}

func testImage(i int) (string, []byte) {
	mimeTypes := []string{"image/png", "image/jpeg", "image/webp"}
	return mimeTypes[i%len(mimeTypes)], []byte(fmt.Sprintf("image-%02d", i))
}

func testImageDataURL(i int) string {
	mimeType, data := testImage(i)
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// assertReferenceImages checks that all count images reached Google in order,
// each with its own MIME type.
func assertReferenceImages(t *testing.T, req map[string]interface{}, count int) {
	t.Helper()
	contents, ok := req["contents"].([]interface{})
	require.True(t, ok)
	require.Len(t, contents, 1)
	parts, ok := contents[0].(map[string]interface{})["parts"].([]interface{})
	require.True(t, ok)
	var images []map[string]interface{}
	for _, part := range parts {
		if inline, ok := part.(map[string]interface{})["inlineData"].(map[string]interface{}); ok {
			images = append(images, inline)
		}
	}
	require.Len(t, images, count)
	for i, image := range images {
		mimeType, data := testImage(i)
		assert.Equal(t, mimeType, image["mimeType"], "image %d", i)
		assert.Equal(t, base64.StdEncoding.EncodeToString(data), image["data"], "image %d", i)
	}
}

func decodeVertexRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))
	return req
}

func generationConfigOf(t *testing.T, req map[string]interface{}) map[string]interface{} {
	t.Helper()
	return mapAt(t, req, "generationConfig")
}

func mapAt(t *testing.T, parent map[string]interface{}, key string) map[string]interface{} {
	t.Helper()
	value, ok := parent[key].(map[string]interface{})
	require.True(t, ok, "%s is not an object: %v", key, parent[key])
	return value
}
