package vertex

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// Vertex AI embedding types (models/{model}:predict)

type VertexEmbeddingRequest struct {
	Instances  []VertexEmbeddingInstance  `json:"instances"`
	Parameters *VertexEmbeddingParameters `json:"parameters,omitempty"`
}

type VertexEmbeddingInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
}

type VertexEmbeddingParameters struct {
	OutputDimensionality *int  `json:"outputDimensionality,omitempty"`
	AutoTruncate         *bool `json:"autoTruncate,omitempty"`
}

type VertexEmbeddingResponse struct {
	Predictions []VertexEmbeddingPrediction `json:"predictions"`
	Metadata    *VertexEmbeddingMetadata    `json:"metadata,omitempty"`
}

type VertexEmbeddingPrediction struct {
	Embeddings VertexEmbeddingValues `json:"embeddings"`
}

type VertexEmbeddingValues struct {
	Values     []float64                  `json:"values"`
	Statistics *VertexEmbeddingStatistics `json:"statistics,omitempty"`
}

type VertexEmbeddingStatistics struct {
	TokenCount float64 `json:"token_count"`
	Truncated  bool    `json:"truncated"`
}

type VertexEmbeddingMetadata struct {
	BillableCharacterCount int `json:"billableCharacterCount,omitempty"`
}

// Gemini API embedding types (models/{model}:batchEmbedContents)

type GeminiEmbeddingRequest struct {
	Requests []GeminiEmbedRequest `json:"requests"`
}

// GeminiEmbedRequest is one Gemini API EmbedContentRequest: the whole body of
// models/{model}:embedContent, or one element of batchEmbedContents' requests.
type GeminiEmbedRequest struct {
	Model                string         `json:"model"`
	Content              *genai.Content `json:"content"`
	TaskType             string         `json:"taskType,omitempty"`
	OutputDimensionality *int32         `json:"outputDimensionality,omitempty"`
}

// GeminiEmbeddingResponse is the raw batchEmbedContents response.
// It matches genai.EmbedContentResponse layout; we use the SDK type directly in
// GeminiEmbeddingToOpenAI so that future SDK additions (e.g. statistics) are
// picked up automatically.

// embedContent types (Gemini Embedding 2 and later)

// VertexEmbedContentRequest is the body of Vertex AI
// models/{model}:embedContent. The endpoint embeds exactly one content per
// call: all of its parts are fused into a single vector.
type VertexEmbedContentRequest struct {
	Content            *genai.Content      `json:"content"`
	EmbedContentConfig *EmbedContentConfig `json:"embedContentConfig,omitempty"`
}

// EmbedContentConfig is the Vertex AI embedContent configuration block.
type EmbedContentConfig struct {
	OutputDimensionality *int32 `json:"outputDimensionality,omitempty"`
}

// EmbeddingUsageMetadata is the usageMetadata of embedContent and
// batchEmbedContents responses. Vertex AI spells the per-modality breakdown
// promptTokensDetails, the Gemini API reference promptTokenDetails; both are read.
type EmbeddingUsageMetadata struct {
	PromptTokenCount    int                           `json:"promptTokenCount,omitempty"`
	TotalTokenCount     int                           `json:"totalTokenCount,omitempty"`
	PromptTokensDetails []EmbeddingModalityTokenCount `json:"promptTokensDetails,omitempty"`
	PromptTokenDetails  []EmbeddingModalityTokenCount `json:"promptTokenDetails,omitempty"`
}

// EmbeddingModalityTokenCount is one modality's share of the prompt tokens.
type EmbeddingModalityTokenCount struct {
	Modality   string `json:"modality,omitempty"`
	TokenCount int    `json:"tokenCount,omitempty"`
}

// embedContentResponse covers every embedContent-family response shape: a
// single embedContent call ("embedding"), Gemini API batchEmbedContents and
// the merged Vertex AI fan-out ("embeddings").
type embedContentResponse struct {
	Embedding     *embedContentValues     `json:"embedding,omitempty"`
	Embeddings    []embedContentValues    `json:"embeddings,omitempty"`
	UsageMetadata *EmbeddingUsageMetadata `json:"usageMetadata,omitempty"`
}

type embedContentValues struct {
	Values []float64 `json:"values"`
}

// MaxEmbedContentInputs caps how many vectors one embedContent-family request
// may ask for. It is the Gemini API batchEmbedContents batch limit, applied to
// the Vertex AI fan-out as well so a request is accepted or refused the same
// way whichever credential serves it.
const MaxEmbedContentInputs = 100

// IsEmbedContentModel reports whether modelID embeds through the multimodal
// embedContent API (Gemini Embedding 2 and later) instead of the legacy
// text-only paths: Vertex AI :predict and the Gemini API batch of text parts.
// gemini-embedding-001 and the retired experimental model stay on those paths.
func IsEmbedContentModel(modelID string) bool {
	model := strings.ToLower(strings.TrimSpace(modelID))
	if slash := strings.LastIndexByte(model, '/'); slash >= 0 {
		model = model[slash+1:]
	}
	if !strings.HasPrefix(model, "gemini-embedding-") {
		return false
	}
	return !strings.HasPrefix(model, "gemini-embedding-001") && !strings.HasPrefix(model, "gemini-embedding-exp")
}

// extractInputTexts parses the OpenAI input field into a slice of strings.
// Handles string, []string, and []interface{} (JSON arrays decode as []interface{}).
func extractInputTexts(input interface{}) ([]string, error) {
	switch v := input.(type) {
	case string:
		return []string{v}, nil
	case []interface{}:
		texts := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, converterutil.NewRequestValidationError("input", fmt.Sprintf("unsupported input array element type: %T; this model accepts only text", item))
			}
			texts = append(texts, s)
		}
		return texts, nil
	case []string:
		return v, nil
	default:
		return nil, converterutil.NewRequestValidationError("input", fmt.Sprintf("unsupported input type: %T", input))
	}
}

// OpenAIEmbeddingToVertex converts an OpenAI embedding request to Vertex AI format.
func OpenAIEmbeddingToVertex(body []byte) ([]byte, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, converterutil.RequestJSONValidationError(err)
	}

	texts, err := extractInputTexts(req.Input)
	if err != nil {
		return nil, err
	}

	instances := make([]VertexEmbeddingInstance, len(texts))
	for i, text := range texts {
		instances[i] = VertexEmbeddingInstance{Content: text}
	}

	vertexReq := VertexEmbeddingRequest{
		Instances: instances,
	}

	if req.Dimensions != nil {
		vertexReq.Parameters = &VertexEmbeddingParameters{
			OutputDimensionality: req.Dimensions,
		}
	}

	return json.Marshal(vertexReq)
}

// OpenAIEmbeddingToGemini converts an OpenAI embedding request to Gemini API format.
func OpenAIEmbeddingToGemini(body []byte, model string) ([]byte, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, converterutil.RequestJSONValidationError(err)
	}

	texts, err := extractInputTexts(req.Input)
	if err != nil {
		return nil, err
	}

	modelRef := "models/" + model

	requests := make([]GeminiEmbedRequest, len(texts))
	for i, text := range texts {
		gr := GeminiEmbedRequest{
			Model: modelRef,
			Content: &genai.Content{
				Parts: []*genai.Part{{Text: text}},
			},
		}
		if req.Dimensions != nil {
			dim := ClampInt32(*req.Dimensions)
			gr.OutputDimensionality = &dim
		}
		requests[i] = gr
	}

	geminiReq := GeminiEmbeddingRequest{
		Requests: requests,
	}

	return json.Marshal(geminiReq)
}

// OpenAIEmbeddingToVertexEmbedContent converts an OpenAI embedding request
// for an embedContent-family model into Vertex AI embedContent bodies, one per
// requested vector and in input order. Vertex AI has no synchronous batch
// method for these models, so each body is a separate upstream call.
func OpenAIEmbeddingToVertexEmbedContent(body []byte) ([][]byte, error) {
	contents, dimensions, err := parseEmbedContentRequest(body)
	if err != nil {
		return nil, err
	}
	var cfg *EmbedContentConfig
	if dimensions != nil {
		cfg = &EmbedContentConfig{OutputDimensionality: dimensions}
	}
	bodies := make([][]byte, len(contents))
	for i, content := range contents {
		encoded, err := json.Marshal(VertexEmbedContentRequest{Content: content, EmbedContentConfig: cfg})
		if err != nil {
			return nil, fmt.Errorf("failed to encode embedContent request: %w", err)
		}
		bodies[i] = encoded
	}
	return bodies, nil
}

// EmbedContentFanOutEnvelope wraps per-input embedContent bodies into one
// {"requests":[...]} document. It is what a fanned-out request is logged as;
// the upstream receives the individual bodies.
func EmbedContentFanOutEnvelope(bodies [][]byte) []byte {
	var buf bytes.Buffer
	buf.WriteString(`{"requests":[`)
	for i, b := range bodies {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(b)
	}
	buf.WriteString(`]}`)
	return buf.Bytes()
}

// OpenAIEmbeddingToGeminiEmbedContent converts an OpenAI embedding request for
// an embedContent-family model into a Gemini API body. A single vector uses
// models/{model}:embedContent; several use the synchronous batchEmbedContents,
// one request per vector. The returned count selects the method (see
// BuildGeminiEmbedContentURL).
func OpenAIEmbeddingToGeminiEmbedContent(body []byte, model string) ([]byte, int, error) {
	contents, dimensions, err := parseEmbedContentRequest(body)
	if err != nil {
		return nil, 0, err
	}
	modelRef := "models/" + model
	requests := make([]GeminiEmbedRequest, len(contents))
	for i, content := range contents {
		requests[i] = GeminiEmbedRequest{Model: modelRef, Content: content, OutputDimensionality: dimensions}
	}
	var encoded []byte
	if len(requests) == 1 {
		encoded, err = json.Marshal(requests[0])
	} else {
		encoded, err = json.Marshal(GeminiEmbeddingRequest{Requests: requests})
	}
	if err != nil {
		return nil, 0, fmt.Errorf("failed to encode embedContent request: %w", err)
	}
	return encoded, len(requests), nil
}

func parseEmbedContentRequest(body []byte) ([]*genai.Content, *int32, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, fmt.Errorf("failed to parse embedding request: %w", err)
	}
	contents, err := parseEmbeddingContents(req.Input)
	if err != nil {
		return nil, nil, err
	}
	var dimensions *int32
	if req.Dimensions != nil {
		if *req.Dimensions <= 0 {
			return nil, nil, converterutil.NewRequestValidationError("dimensions", "dimensions must be a positive integer")
		}
		dim := ClampInt32(*req.Dimensions)
		dimensions = &dim
	}
	return contents, dimensions, nil
}

// parseEmbeddingContents converts the OpenAI input field of an
// embedContent-family request into one genai.Content per requested vector,
// in input order. The top level follows OpenAI: a single item, or an array
// with one item per vector. An item is
//
//   - a string: text;
//   - a content part object ({"type": "text" | "image_url" | "input_audio" |
//     "video_url" | "file", ...}, the chat content part shapes);
//   - an array of strings and content parts, or {"content": [...]}: all its
//     parts are fused into one vector (mixed input).
//
// Anything that cannot be turned into a part is refused with a 400 rather than
// dropped, since a silently shortened input still yields a vector and bills.
func parseEmbeddingContents(input interface{}) ([]*genai.Content, error) {
	var items []interface{}
	switch v := input.(type) {
	case nil:
		return nil, converterutil.NewRequestValidationError("input", "input is required")
	case []interface{}:
		if len(v) == 0 {
			return nil, converterutil.NewRequestValidationError("input", "input must not be empty")
		}
		items = v
	default:
		items = []interface{}{v}
	}
	if len(items) > MaxEmbedContentInputs {
		return nil, converterutil.NewRequestValidationError("input",
			fmt.Sprintf("at most %d inputs per request are supported for this model, got %d", MaxEmbedContentInputs, len(items)))
	}
	contents := make([]*genai.Content, len(items))
	for i, item := range items {
		parts, err := embeddingItemParts(item, fmt.Sprintf("input[%d]", i))
		if err != nil {
			return nil, err
		}
		contents[i] = &genai.Content{Parts: parts}
	}
	return contents, nil
}

func embeddingItemParts(item interface{}, param string) ([]*genai.Part, error) {
	switch v := item.(type) {
	case string:
		return []*genai.Part{{Text: v}}, nil
	case []interface{}:
		return embeddingPartList(v, param)
	case map[string]interface{}:
		if _, ok := v["type"]; ok {
			part, err := embeddingPart(v, param)
			if err != nil {
				return nil, err
			}
			return []*genai.Part{part}, nil
		}
		if content, ok := v["content"]; ok {
			switch c := content.(type) {
			case string:
				return []*genai.Part{{Text: c}}, nil
			case []interface{}:
				return embeddingPartList(c, param+".content")
			}
			return nil, converterutil.NewRequestValidationError(param+".content", "content must be a string or an array of content parts")
		}
		return nil, converterutil.NewRequestValidationError(param, `an input object must be a content part with "type" or carry "content"`)
	case float64:
		return nil, converterutil.NewRequestValidationError(param, "token id arrays are not supported for this model; send text")
	default:
		return nil, converterutil.NewRequestValidationError(param, fmt.Sprintf("unsupported input element type: %T", item))
	}
}

func embeddingPartList(list []interface{}, param string) ([]*genai.Part, error) {
	if len(list) == 0 {
		return nil, converterutil.NewRequestValidationError(param, "content parts must not be empty")
	}
	parts := make([]*genai.Part, 0, len(list))
	for i, element := range list {
		elementParam := fmt.Sprintf("%s[%d]", param, i)
		switch e := element.(type) {
		case string:
			parts = append(parts, &genai.Part{Text: e})
		case map[string]interface{}:
			part, err := embeddingPart(e, elementParam)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case float64:
			return nil, converterutil.NewRequestValidationError(elementParam, "token id arrays are not supported for this model; send text")
		default:
			return nil, converterutil.NewRequestValidationError(elementParam, fmt.Sprintf("unsupported content part type: %T", element))
		}
	}
	return parts, nil
}

// embeddingPart converts one chat-style content part into a genai.Part: text,
// inline data (MIME type + bytes from a data: URL or base64) or a file
// reference (MIME type + https:// or gs:// URI the provider fetches itself).
func embeddingPart(block map[string]interface{}, param string) (*genai.Part, error) {
	partType, _ := block["type"].(string)
	switch partType {
	case "text", "input_text":
		text, ok := block["text"].(string)
		if !ok {
			return nil, converterutil.NewRequestValidationError(param+".text", "text part requires a string text field")
		}
		return &genai.Part{Text: text}, nil
	case "image_url", "input_image":
		ref, obj := mediaReference(block, "image_url", "url")
		return embeddingMediaPart(ref, mimeOverride(obj, block), "", param+".image_url")
	case "video_url", "input_video":
		ref, obj := mediaReference(block, "video_url", "url")
		return embeddingMediaPart(ref, mimeOverride(obj, block), "video/mp4", param+".video_url")
	case "input_audio":
		audio, _ := block["input_audio"].(map[string]interface{})
		if audio == nil {
			return nil, converterutil.NewRequestValidationError(param+".input_audio", "input_audio part requires an input_audio object")
		}
		mimeType := "audio/wav"
		if format, _ := audio["format"].(string); format != "" {
			mimeType = getAudioMimeType(format)
		}
		if override := mimeOverride(audio, block); override != "" {
			mimeType = override
		}
		if data, _ := audio["data"].(string); data != "" {
			return inlineBase64Part(data, mimeType, param+".input_audio.data")
		}
		if ref, _ := audio["url"].(string); ref != "" {
			return embeddingMediaPart(ref, mimeType, "", param+".input_audio.url")
		}
		return nil, converterutil.NewRequestValidationError(param+".input_audio", "input_audio requires base64 data")
	case "file", "input_file":
		file, _ := block["file"].(map[string]interface{})
		if file == nil {
			file = block
		}
		override := mimeOverride(file, block)
		for _, key := range []string{"file_data", "file_url", "url", "file_id"} {
			ref, _ := file[key].(string)
			if ref == "" {
				continue
			}
			if key == "file_data" && !strings.HasPrefix(ref, "data:") {
				// OpenAI chat file_data may be bare base64; the MIME type then
				// has to come from mime_type/format or the file name.
				if override == "" {
					if name, _ := file["filename"].(string); name != "" {
						override = mimeTypeFromName(name)
					}
				}
				if override == "" {
					return nil, converterutil.NewRequestValidationError(param+".file.file_data", "bare base64 file_data requires mime_type (or a data: URL)")
				}
				return inlineBase64Part(ref, override, param+".file.file_data")
			}
			return embeddingMediaPart(ref, override, "", param+".file."+key)
		}
		return nil, converterutil.NewRequestValidationError(param+".file", "file part requires file_data (data: URL or base64) or file_url (https:// or gs://)")
	case "":
		return nil, converterutil.NewRequestValidationError(param+".type", "content part type is required")
	default:
		return nil, converterutil.NewRequestValidationError(param+".type", fmt.Sprintf("unsupported content part type %q", partType))
	}
}

// mediaReference reads the URL of an image_url/video_url style part, which is
// either {"image_url": {"url": "..."}} or {"image_url": "..."}.
func mediaReference(block map[string]interface{}, field, urlKey string) (string, map[string]interface{}) {
	switch v := block[field].(type) {
	case string:
		return v, nil
	case map[string]interface{}:
		ref, _ := v[urlKey].(string)
		return ref, v
	}
	return "", nil
}

// mimeOverride returns an explicit MIME type from mime_type or format on the
// media object or the part itself. format may be a MIME type or an extension.
func mimeOverride(objects ...map[string]interface{}) string {
	for _, obj := range objects {
		if obj == nil {
			continue
		}
		for _, key := range []string{"mime_type", "mimeType", "format"} {
			value, _ := obj[key].(string)
			value = strings.TrimSpace(value)
			if value == "" {
				continue
			}
			if strings.Contains(value, "/") {
				return value
			}
			if mimeType := mimeTypeFromName("file." + value); mimeType != "" {
				return mimeType
			}
		}
	}
	return ""
}

// embeddingAudioExtensions extends mimeTypeMap with the audio containers
// Gemini Embedding 2 accepts; kept local so chat URL handling is unchanged.
var embeddingAudioExtensions = map[string]string{
	"mp3": "audio/mpeg",
	"wav": "audio/wav",
}

func mimeTypeFromName(name string) string {
	if mimeType := getMimeTypeFromURL(name); mimeType != "" {
		return mimeType
	}
	ext := ""
	if idx := strings.LastIndex(name, "."); idx >= 0 {
		ext = strings.ToLower(name[idx+1:])
	}
	return embeddingAudioExtensions[ext]
}

func embeddingMediaPart(ref, override, defaultMIME, param string) (*genai.Part, error) {
	if ref == "" {
		return nil, converterutil.NewRequestValidationError(param, "media part requires a url")
	}
	if strings.HasPrefix(ref, "data:") {
		part, err := parseDataURLToPart(ref)
		if err != nil {
			return nil, err
		}
		if part == nil {
			return nil, converterutil.NewRequestValidationError(param, "invalid or empty base64 data URL")
		}
		if override != "" {
			part.InlineData.MIMEType = override
		}
		return part, nil
	}
	mimeType := override
	if mimeType == "" {
		mimeType = mimeTypeFromName(stripURLQuery(ref))
	}
	if mimeType == "" {
		mimeType = defaultMIME
	}
	if mimeType == "" {
		return nil, converterutil.NewRequestValidationError(param, "cannot determine the MIME type of the file; set mime_type")
	}
	part := parseURLToPart(ref, map[string]interface{}{"format": mimeType})
	if part == nil {
		return nil, converterutil.NewRequestValidationError(param, "unsupported URL: use a data: URL, a public https:// URL or a gs:// URI")
	}
	return part, nil
}

func stripURLQuery(ref string) string {
	if idx := strings.IndexAny(ref, "?#"); idx >= 0 {
		return ref[:idx]
	}
	return ref
}

func inlineBase64Part(data, mimeType, param string) (*genai.Part, error) {
	if len(data) > maxBase64Size {
		return nil, converterutil.NewRequestEntityTooLargeError(param, fmt.Sprintf("inline payload exceeds %dMB limit", maxBase64Size/(1024*1024)))
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, converterutil.NewRequestValidationError(param, "data is not valid base64")
	}
	if len(decoded) == 0 {
		return nil, converterutil.NewRequestValidationError(param, "data is empty")
	}
	return &genai.Part{InlineData: &genai.Blob{MIMEType: mimeType, Data: decoded}}, nil
}

// VertexEmbeddingToOpenAI converts a Vertex AI embedding response to OpenAI format.
func VertexEmbeddingToOpenAI(body []byte, model string) ([]byte, error) {
	var resp VertexEmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse vertex embedding response: %w", err)
	}

	data := make([]openai.OpenAIEmbeddingData, len(resp.Predictions))
	totalTokens := 0

	for i, pred := range resp.Predictions {
		data[i] = openai.OpenAIEmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: pred.Embeddings.Values,
		}
		if pred.Embeddings.Statistics != nil {
			totalTokens += int(math.Round(pred.Embeddings.Statistics.TokenCount))
		}
	}

	openaiResp := openai.OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: openai.OpenAIEmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	return json.Marshal(openaiResp)
}

// estimateTokens returns a rough token count for text using the ~4 chars/token heuristic.
func estimateTokens(text string) int {
	n := len([]rune(text))
	if n == 0 {
		return 1
	}
	t := (n + 3) / 4
	if t < 1 {
		t = 1
	}
	return t
}

// ExtractEmbeddingTexts parses an OpenAI embedding request body and returns the input texts.
// Used to cache texts so GeminiEmbeddingToOpenAI can estimate prompt_tokens when
// batchEmbedContents omits usage (legacy text-only models only).
func ExtractEmbeddingTexts(body []byte) ([]string, error) {
	var req openai.OpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("failed to parse embedding request: %w", err)
	}
	return extractInputTexts(req.Input)
}

// geminiBatchEmbeddingResponse is the legacy batchEmbedContents response:
// the genai SDK layout plus the usageMetadata the SDK type does not carry.
type geminiBatchEmbeddingResponse struct {
	genai.EmbedContentResponse
	UsageMetadata *EmbeddingUsageMetadata `json:"usageMetadata,omitempty"`
}

// GeminiEmbeddingToOpenAI converts a Gemini API batchEmbedContents response to OpenAI format.
// Token usage comes from usageMetadata, then per-embedding statistics; only when
// the response carries neither is it estimated from inputTexts using a ~4
// chars/token heuristic (legacy text-only models; embedContent-family models
// go through EmbedContentToOpenAI).
func GeminiEmbeddingToOpenAI(body []byte, model string, inputTexts []string) ([]byte, error) {
	var resp geminiBatchEmbeddingResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse gemini embedding response: %w", err)
	}

	data := make([]openai.OpenAIEmbeddingData, len(resp.Embeddings))
	var promptTokens int
	for i, emb := range resp.Embeddings {
		if emb == nil {
			emb = &genai.ContentEmbedding{}
		}
		// genai uses float32; OpenAI expects float64.
		embedding := make([]float64, len(emb.Values))
		for j, v := range emb.Values {
			embedding[j] = float64(v)
		}
		data[i] = openai.OpenAIEmbeddingData{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
		}
		if emb.Statistics != nil && emb.Statistics.TokenCount > 0 {
			promptTokens += int(math.Round(float64(emb.Statistics.TokenCount)))
		}
	}

	usage := openai.OpenAIEmbeddingUsage{PromptTokens: promptTokens, TotalTokens: promptTokens}
	if resp.UsageMetadata != nil {
		if metered := resp.UsageMetadata.openAIUsage(); metered.PromptTokens > 0 {
			usage = metered
		}
	}

	// Fallback: the response reported no usage at all.
	// Estimate from the original request texts (~4 chars per token).
	if usage.PromptTokens == 0 {
		for _, text := range inputTexts {
			usage.PromptTokens += estimateTokens(text)
		}
		usage.TotalTokens = usage.PromptTokens
	}

	openaiResp := openai.OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage:  usage,
	}

	return json.Marshal(openaiResp)
}

// EmbedContentToOpenAI converts an embedContent-family response — a single
// embedContent call, Gemini API batchEmbedContents, or the merged Vertex AI
// fan-out (MergeEmbedContentResponses) — to OpenAI embeddings format. Vectors
// keep request order and index; usage comes from usageMetadata, split by
// modality in prompt_tokens_details. Embedding vectors are not output tokens.
//
// A reply without usage would otherwise bill nothing, so usage is then
// estimated from the text parts of request (the OpenAI request body) and
// estimated is true: the caller should log it, since media parts cannot be
// sized from the request and stay unbilled.
func EmbedContentToOpenAI(body []byte, model string, request []byte) (converted []byte, estimated bool, err error) {
	var resp embedContentResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, false, fmt.Errorf("failed to parse embedContent response: %w", err)
	}
	vectors := resp.Embeddings
	if resp.Embedding != nil {
		vectors = append([]embedContentValues{*resp.Embedding}, vectors...)
	}
	if len(vectors) == 0 {
		return nil, false, fmt.Errorf("embedContent response carries no embeddings")
	}

	data := make([]openai.OpenAIEmbeddingData, len(vectors))
	for i, vector := range vectors {
		values := vector.Values
		if values == nil {
			values = []float64{}
		}
		data[i] = openai.OpenAIEmbeddingData{Object: "embedding", Index: i, Embedding: values}
	}

	var usage openai.OpenAIEmbeddingUsage
	if resp.UsageMetadata != nil {
		usage = resp.UsageMetadata.openAIUsage()
	}
	if usage.PromptTokens == 0 {
		usage = estimateEmbedContentTextUsage(request)
		estimated = true
	}
	converted, err = json.Marshal(openai.OpenAIEmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage:  usage,
	})
	return converted, estimated, err
}

// estimateEmbedContentTextUsage sizes the text parts of an embedContent-family
// OpenAI request at ~4 characters per token, the estimate the legacy text-only
// path uses. It is zero when request is empty or cannot be parsed.
func estimateEmbedContentTextUsage(request []byte) openai.OpenAIEmbeddingUsage {
	var req openai.OpenAIEmbeddingRequest
	if len(request) == 0 || json.Unmarshal(request, &req) != nil {
		return openai.OpenAIEmbeddingUsage{}
	}
	contents, err := parseEmbeddingContents(req.Input)
	if err != nil {
		return openai.OpenAIEmbeddingUsage{}
	}
	tokens := 0
	for _, content := range contents {
		for _, part := range content.Parts {
			if part.Text != "" {
				tokens += estimateTokens(part.Text)
			}
		}
	}
	return openai.OpenAIEmbeddingUsage{PromptTokens: tokens, TotalTokens: tokens}
}

// MergeEmbedContentResponses joins single-content Vertex AI embedContent
// responses, given in input order, into the batch shape EmbedContentToOpenAI
// reads: {"embeddings": [...], "usageMetadata": {...}} with usage summed per
// modality.
func MergeEmbedContentResponses(bodies [][]byte) ([]byte, error) {
	type singleResponse struct {
		Embedding     json.RawMessage         `json:"embedding"`
		UsageMetadata *EmbeddingUsageMetadata `json:"usageMetadata,omitempty"`
	}
	merged := struct {
		Embeddings    []json.RawMessage       `json:"embeddings"`
		UsageMetadata *EmbeddingUsageMetadata `json:"usageMetadata,omitempty"`
	}{Embeddings: make([]json.RawMessage, len(bodies))}

	var usage EmbeddingUsageMetadata
	hasUsage := false
	byModality := make(map[string]int)
	var modalityOrder []string
	for i, body := range bodies {
		var single singleResponse
		if err := json.Unmarshal(body, &single); err != nil {
			return nil, fmt.Errorf("embedContent response %d: %w", i, err)
		}
		if len(single.Embedding) == 0 || string(single.Embedding) == "null" {
			return nil, fmt.Errorf("embedContent response %d carries no embedding", i)
		}
		merged.Embeddings[i] = single.Embedding
		if single.UsageMetadata == nil {
			continue
		}
		hasUsage = true
		usage.PromptTokenCount += single.UsageMetadata.PromptTokenCount
		usage.TotalTokenCount += single.UsageMetadata.TotalTokenCount
		for _, detail := range single.UsageMetadata.modalityDetails() {
			if _, seen := byModality[detail.Modality]; !seen {
				modalityOrder = append(modalityOrder, detail.Modality)
			}
			byModality[detail.Modality] += detail.TokenCount
		}
	}
	if hasUsage {
		for _, modality := range modalityOrder {
			usage.PromptTokensDetails = append(usage.PromptTokensDetails, EmbeddingModalityTokenCount{
				Modality:   modality,
				TokenCount: byModality[modality],
			})
		}
		merged.UsageMetadata = &usage
	}
	return json.Marshal(merged)
}

func (m *EmbeddingUsageMetadata) modalityDetails() []EmbeddingModalityTokenCount {
	if len(m.PromptTokensDetails) > 0 {
		return m.PromptTokensDetails
	}
	return m.PromptTokenDetails
}

// openAIUsage maps usageMetadata to OpenAI embeddings usage. IMAGE and
// DOCUMENT (PDF pages are rendered and counted as images) become image
// tokens, AUDIO audio tokens, VIDEO video tokens; TEXT and any unrecognised
// modality stay in the text remainder of prompt_tokens.
func (m *EmbeddingUsageMetadata) openAIUsage() openai.OpenAIEmbeddingUsage {
	var details openai.OpenAIEmbeddingPromptDetails
	sum := 0
	for _, detail := range m.modalityDetails() {
		count := converterutil.NonNegativeTokenCount(detail.TokenCount)
		sum += count
		switch strings.ToUpper(detail.Modality) {
		case string(genai.MediaModalityImage), string(genai.MediaModalityDocument):
			details.ImageTokens += count
		case string(genai.MediaModalityAudio):
			details.AudioTokens += count
		case string(genai.MediaModalityVideo):
			details.VideoTokens += count
		default:
			details.TextTokens += count
		}
	}
	promptTokens := converterutil.NonNegativeTokenCount(m.PromptTokenCount)
	if promptTokens == 0 {
		promptTokens = converterutil.NonNegativeTokenCount(m.TotalTokenCount)
	}
	if sum > promptTokens {
		promptTokens = sum
	}
	if sum > 0 && sum < promptTokens {
		details.TextTokens += promptTokens - sum
	}
	usage := openai.OpenAIEmbeddingUsage{PromptTokens: promptTokens, TotalTokens: promptTokens}
	if details != (openai.OpenAIEmbeddingPromptDetails{}) {
		usage.PromptTokensDetails = &details
	}
	return usage
}

// BuildVertexEmbeddingURL constructs the Vertex AI URL for embeddings.
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexEmbeddingURL(cred *config.CredentialConfig, modelID string) string {
	return buildVertexEmbeddingMethodURL(cred, modelID, "predict")
}

// BuildVertexEmbedContentURL constructs the Vertex AI embedContent URL used
// by embedContent-family models (see IsEmbedContentModel). Gemini Embedding 2
// quotas are global, so credentials normally use location "global".
func BuildVertexEmbedContentURL(cred *config.CredentialConfig, modelID string) string {
	return buildVertexEmbeddingMethodURL(cred, modelID, "embedContent")
}

func buildVertexEmbeddingMethodURL(cred *config.CredentialConfig, modelID, method string) string {
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/google/models/%s:%s",
			cred.ProjectID, modelID, method,
		)
	}

	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		cred.Location, cred.ProjectID, cred.Location, modelID, method,
	)
}

// BuildGeminiEmbeddingURL constructs the Gemini API URL for embeddings.
// Format: {base_url}/v1beta/models/{model}:batchEmbedContents
func BuildGeminiEmbeddingURL(cred *config.CredentialConfig, modelID string) string {
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	return fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents", baseURL, modelID)
}

// BuildGeminiEmbedContentURL constructs the Gemini API URL for an
// embedContent-family request: models/{model}:embedContent for one vector,
// the synchronous batchEmbedContents for several.
func BuildGeminiEmbedContentURL(cred *config.CredentialConfig, modelID string, inputs int) string {
	if inputs > 1 {
		return BuildGeminiEmbeddingURL(cred, modelID)
	}
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")
	return fmt.Sprintf("%s/v1beta/models/%s:embedContent", baseURL, modelID)
}
