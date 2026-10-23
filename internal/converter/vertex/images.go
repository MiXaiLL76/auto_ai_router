package vertex

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"

	"google.golang.org/genai"
)

// VertexImageRequest represents Vertex AI Imagen request
type VertexImageRequest struct {
	Instances  []VertexImageInstance `json:"instances"`
	Parameters VertexImageParameters `json:"parameters"`
}

type VertexImageInstance struct {
	Prompt string `json:"prompt"`
}

type VertexImageParameters struct {
	SampleCount       int    `json:"sampleCount,omitempty"`
	AspectRatio       string `json:"aspectRatio,omitempty"`
	SafetyFilterLevel string `json:"safetyFilterLevel,omitempty"`
	PersonGeneration  string `json:"personGeneration,omitempty"`
}

// VertexImageResponse represents Vertex AI Imagen response
type VertexImageResponse struct {
	Predictions []VertexImagePrediction `json:"predictions"`
}

type VertexImagePrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}

// BuildVertexImageURL constructs the Vertex AI URL for image generation
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexImageURL(cred *config.CredentialConfig, modelID string) string {
	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// OpenAIImageToVertex converts OpenAI image request to Vertex AI Imagen format
func OpenAIImageToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, converterutil.RequestJSONValidationError(err)
	}

	// Convert size to aspect ratio
	aspectRatio := "1:1" // default
	switch openAIReq.Size {
	case "1024x1024", "512x512", "256x256":
		aspectRatio = "1:1"
	case "1792x1024":
		aspectRatio = "16:9"
	case "1024x1792":
		aspectRatio = "9:16"
	}

	// Set sample count (max 10 for image generation)
	sampleCount := 1
	if openAIReq.N != nil && *openAIReq.N > 0 {
		sampleCount = *openAIReq.N
		if sampleCount > 10 {
			sampleCount = 10
		}
	}

	// Handle quality and style (basic mapping)
	safetyLevel := "block_some"
	if openAIReq.Quality == "hd" {
		// For HD quality, we might want stricter safety
		safetyLevel = "block_few"
	}

	vertexReq := VertexImageRequest{
		Instances: []VertexImageInstance{
			{Prompt: openAIReq.Prompt},
		},
		Parameters: VertexImageParameters{
			SampleCount:       sampleCount,
			AspectRatio:       aspectRatio,
			SafetyFilterLevel: safetyLevel,
			PersonGeneration:  "allow_adult",
		},
	}

	return json.Marshal(vertexReq)
}

// VertexImageToOpenAI converts Vertex AI Imagen response to OpenAI format
func VertexImageToOpenAI(vertexBody []byte) ([]byte, error) {
	var vertexResp VertexImageResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex image response: %w", err)
	}

	openAIResp := openai.OpenAIImageResponse{
		Created: converterutil.GetCurrentTimestamp(),
		Data:    make([]openai.OpenAIImageData, 0),
	}

	// Convert predictions to OpenAI format
	for _, prediction := range vertexResp.Predictions {
		data := openai.OpenAIImageData{
			B64JSON: prediction.BytesBase64Encoded,
		}
		openAIResp.Data = append(openAIResp.Data, data)
	}

	return json.Marshal(openAIResp)
}

// ImageRequestToOpenAIChatRequest converts OpenAI image generation request to OpenAI chat request format
// This allows Gemini models to generate images through chat API with response_modalities: ["IMAGE"]
func ImageRequestToOpenAIChatRequest(openAIBody []byte) ([]byte, error) {
	return imageRequestToOpenAIChatRequest(openAIBody, "")
}

func imageRequestToOpenAIChatRequest(openAIBody []byte, providerModel string) ([]byte, error) {
	var body struct {
		openai.OpenAIImageRequest
		geminiImageExtras
	}
	if err := json.Unmarshal(openAIBody, &body); err != nil {
		return nil, converterutil.RequestJSONValidationError(err)
	}
	imageReq, extras := body.OpenAIImageRequest, body.geminiImageExtras

	if strings.TrimSpace(imageReq.Prompt) == "" {
		return nil, imageValidationError("prompt", "Missing required parameter", "missing_required_parameter")
	}
	if imageReq.N != nil && *imageReq.N <= 0 {
		return nil, converterutil.NewInvalidValueError("n")
	}

	genConfig := map[string]interface{}{
		"response_modalities": []string{"IMAGE"},
	}
	if providerModel == "" {
		providerModel = imageReq.Model
	}
	if err := applyGeminiImageSize(genConfig, providerModel, imageReq.Size); err != nil {
		return nil, err
	}

	// Convert to OpenAI chat request format
	chatReq := openai.OpenAIRequest{
		Model:       imageReq.Model,
		Temperature: imageReq.Temperature,
		TopP:        imageReq.TopP,
		Messages: []openai.OpenAIMessage{
			{
				Role:    "user",
				Content: imageReq.Prompt,
			},
		},
		ExtraBody: map[string]interface{}{
			"generation_config": genConfig,
		},
	}
	if imageReq.Seed != nil {
		if *imageReq.Seed < math.MinInt32 || *imageReq.Seed > math.MaxInt32 {
			return nil, converterutil.NewInvalidValueError("seed")
		}
		chatReq.Seed = imageReq.Seed
	}
	if imageReq.N != nil && *imageReq.N > 0 {
		n := clampImageCount(*imageReq.N)
		chatReq.N = &n
	}
	if err := extras.applyTo(&chatReq, providerModel); err != nil {
		return nil, err
	}

	return json.Marshal(chatReq)
}

// geminiImageExtras are the Gemini parameters every images endpoint accepts and hands
// to the chat request's fields of the same names. image_config is an object or that
// object as a JSON string (the multipart form); the others are strings.
//
// A JSON body used to pass them unread, whatever their type. So they are decoded raw
// and checked in applyTo: there a value of the wrong type is a 400 only for a
// profiled model (whose parameters are documented) and is ignored for any other. The
// multipart form has always rejected a malformed image_config, for every model.
type geminiImageExtras struct {
	ImageConfig        json.RawMessage `json:"image_config"`
	ImageConfigCamel   json.RawMessage `json:"imageConfig"`
	AspectRatio        json.RawMessage `json:"aspect_ratio"`
	AspectRatioCamel   json.RawMessage `json:"aspectRatio"`
	ImageSize          json.RawMessage `json:"image_size"`
	ImageSizeCamel     json.RawMessage `json:"imageSize"`
	ThinkingLevel      json.RawMessage `json:"thinking_level"`
	ThinkingLevelCamel json.RawMessage `json:"thinkingLevel"`
	ReasoningEffort    json.RawMessage `json:"reasoning_effort"`

	fromForm bool // read from multipart form fields
}

// geminiImageExtrasFromFields reads the extras from multipart form fields.
func geminiImageExtrasFromFields(fields map[string]string) geminiImageExtras {
	rawJSON := func(name string) json.RawMessage {
		if fields[name] == "" {
			return nil
		}
		return json.RawMessage(strconv.Quote(fields[name]))
	}
	return geminiImageExtras{
		fromForm:           true,
		ImageConfig:        rawJSON("image_config"),
		ImageConfigCamel:   rawJSON("imageConfig"),
		AspectRatio:        rawJSON("aspect_ratio"),
		AspectRatioCamel:   rawJSON("aspectRatio"),
		ImageSize:          rawJSON("image_size"),
		ImageSizeCamel:     rawJSON("imageSize"),
		ThinkingLevel:      rawJSON("thinking_level"),
		ThinkingLevelCamel: rawJSON("thinkingLevel"),
		ReasoningEffort:    rawJSON("reasoning_effort"),
	}
}

// applyTo sets the extras on the chat request, where they override what size mapped
// to. Thinking fields pass only to a model with profiled thinking levels.
func (e geminiImageExtras) applyTo(chatReq *openai.OpenAIRequest, model string) error {
	profile := lookupGeminiModelProfile(model)
	p := imageExtraParser{strict: e.fromForm || profile != nil}
	chatReq.ImageConfigSnake = p.objectField("image_config", e.ImageConfig)
	chatReq.ImageConfig = p.objectField("imageConfig", e.ImageConfigCamel)
	chatReq.AspectRatioSnake = p.stringField("aspect_ratio", e.AspectRatio)
	chatReq.AspectRatio = p.stringField("aspectRatio", e.AspectRatioCamel)
	chatReq.ImageSizeSnake = p.stringField("image_size", e.ImageSize)
	chatReq.ImageSize = p.stringField("imageSize", e.ImageSizeCamel)
	if profile.hasThinkingLevels() {
		chatReq.ThinkingLevel = p.stringField("thinking_level", e.ThinkingLevel)
		chatReq.ThinkingLevelCamel = p.stringField("thinkingLevel", e.ThinkingLevelCamel)
		chatReq.ReasoningEffort = p.stringField("reasoning_effort", e.ReasoningEffort)
	}
	return p.err
}

// imageExtraParser decodes geminiImageExtras values. A value it cannot use is
// dropped; when strict, the first such value is also kept as err.
type imageExtraParser struct {
	strict bool
	err    error
}

func (p *imageExtraParser) fail(err error) {
	if p.strict && p.err == nil {
		p.err = err
	}
}

// stringField reads a JSON string; absent and null are "".
func (p *imageExtraParser) stringField(name string, raw json.RawMessage) string {
	if isAbsentJSON(raw) {
		return ""
	}
	var value string
	if json.Unmarshal(raw, &value) != nil {
		p.fail(converterutil.NewInvalidTypeError(name))
		return ""
	}
	return value
}

// objectField reads an object or a JSON string of one; absent, null and "" are nil.
func (p *imageExtraParser) objectField(name string, raw json.RawMessage) map[string]interface{} {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		raw = json.RawMessage(text)
	}
	if isAbsentJSON(raw) {
		return nil
	}
	var parsed map[string]interface{}
	if json.Unmarshal(raw, &parsed) != nil {
		p.fail(imageValidationError(name, "Invalid JSON", "invalid_json"))
		return nil
	}
	return parsed
}

func isAbsentJSON(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) == 0 || string(raw) == "null"
}

// ImageEditRequestToOpenAIChatRequest converts JSON or multipart image edit requests
// to an OpenAI chat request for Gemini image-capable models.
func ImageEditRequestToOpenAIChatRequest(openAIBody []byte, contentType string) ([]byte, error) {
	return imageEditRequestToOpenAIChatRequest(openAIBody, contentType, "")
}

func imageEditRequestToOpenAIChatRequest(openAIBody []byte, contentType, providerModel string) ([]byte, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, converterutil.NewInvalidValueError("content_type")
	}
	if mediaType == "application/json" {
		return imageEditJSONToOpenAIChatRequest(openAIBody, providerModel)
	}
	if mediaType != "multipart/form-data" {
		return nil, &converterutil.RequestValidationError{Param: "content_type", Message: "Image edits require JSON or multipart form data", StatusCode: http.StatusUnsupportedMediaType}
	}

	boundary := params["boundary"]
	if boundary == "" {
		return nil, converterutil.NewInvalidValueError("content_type")
	}

	reader := multipart.NewReader(bytes.NewReader(openAIBody), boundary)
	fields := make(map[string]string)
	contentBlocks := make([]map[string]interface{}, 0)
	maskBlocks := make([]map[string]interface{}, 0)

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, imageValidationError("image", "Invalid multipart form data", "invalid_multipart")
		}

		formName := part.FormName()
		if formName == "" {
			continue
		}

		data, readErr := readMultipartPartLimit(part, 20*1024*1024)
		if readErr != nil {
			return nil, readErr
		}

		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}

		mimeType, mimeErr := detectImageMIMEType(part.Header.Get("Content-Type"), data)
		if mimeErr != nil {
			return nil, imageValidationError(formName, "Invalid image data", "invalid_image")
		}

		block := map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data),
			},
		}

		switch formName {
		case "image", "images", "image[]":
			contentBlocks = append(contentBlocks, block)
		case "mask":
			maskBlocks = append(maskBlocks, block)
		}
	}

	model := strings.TrimSpace(fields["model"])
	if model == "" {
		return nil, imageValidationError("model", "Missing required parameter", "missing_required_parameter")
	}
	prompt := strings.TrimSpace(fields["prompt"])
	if prompt == "" {
		return nil, imageValidationError("prompt", "Missing required parameter", "missing_required_parameter")
	}

	if len(contentBlocks) == 0 {
		return nil, imageValidationError("image", "Missing required parameter", "missing_required_parameter")
	}

	messageBlocks := make([]map[string]interface{}, 0, 1+len(contentBlocks)+len(maskBlocks))
	messageBlocks = append(messageBlocks, map[string]interface{}{
		"type": "text",
		"text": prompt,
	})
	messageBlocks = append(messageBlocks, contentBlocks...)
	if len(maskBlocks) > 0 {
		messageBlocks = append(messageBlocks, map[string]interface{}{
			"type": "text",
			"text": "Use the provided mask image to constrain the edit.",
		})
		messageBlocks = append(messageBlocks, maskBlocks...)
	}

	genConfig := map[string]interface{}{
		"response_modalities": []string{"IMAGE"},
	}
	if providerModel == "" {
		providerModel = model
	}
	if err := applyGeminiImageSize(genConfig, providerModel, fields["size"]); err != nil {
		return nil, err
	}

	chatReq := openai.OpenAIRequest{
		Model: model,
		Messages: []openai.OpenAIMessage{
			{
				Role:    "user",
				Content: messageBlocks,
			},
		},
		ExtraBody: map[string]interface{}{
			"generation_config": genConfig,
		},
	}
	if rawSeed := strings.TrimSpace(fields["seed"]); rawSeed != "" {
		seed, err := strconv.ParseInt(rawSeed, 10, 32)
		if err != nil {
			return nil, converterutil.NewInvalidValueError("seed")
		}
		chatReq.Seed = &seed
	}
	if rawTemperature := strings.TrimSpace(fields["temperature"]); rawTemperature != "" {
		temperature, err := parseImageEditFloat(rawTemperature, "temperature")
		if err != nil {
			return nil, err
		}
		chatReq.Temperature = &temperature
	}
	if rawTopP := strings.TrimSpace(fields["top_p"]); rawTopP != "" {
		topP, err := parseImageEditFloat(rawTopP, "top_p")
		if err != nil {
			return nil, err
		}
		chatReq.TopP = &topP
	}
	if rawN := strings.TrimSpace(fields["n"]); rawN != "" {
		n, err := strconv.Atoi(rawN)
		if err != nil {
			return nil, converterutil.NewInvalidTypeError("n")
		}
		if n <= 0 {
			return nil, converterutil.NewInvalidValueError("n")
		}
		n = clampImageCount(n)
		chatReq.N = &n
	}
	if err := geminiImageExtrasFromFields(fields).applyTo(&chatReq, providerModel); err != nil {
		return nil, err
	}

	return json.Marshal(chatReq)
}

// VertexChatResponseToOpenAIImage converts Vertex AI chat response with image to OpenAI image format
// Extracts inline image data from chat response and returns it in OpenAI image generation format
func VertexChatResponseToOpenAIImage(vertexBody []byte) ([]byte, error) {
	return VertexChatResponseToOpenAIImageWithModel(vertexBody, "")
}

// VertexChatResponseToOpenAIImageWithModel preserves the client-visible model
// on transformed Gemini Images responses.
func VertexChatResponseToOpenAIImageWithModel(vertexBody []byte, model string) ([]byte, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex chat response: %w", err)
	}

	openAIResp := openai.OpenAIImageResponse{
		Created: converterutil.GetCurrentTimestamp(),
		Data:    make([]openai.OpenAIImageData, 0),
		Model:   model,
	}

	// Extract images from candidates
	for _, candidate := range vertexResp.Candidates {
		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
				// A thought image is an interim draft, not a result.
				if part.Thought {
					continue
				}
				// Extract inline data (image) from part
				if part.InlineData != nil {
					// Encode binary image data to base64
					b64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
					imageData := openai.OpenAIImageData{
						B64JSON: b64Data,
					}
					openAIResp.Data = append(openAIResp.Data, imageData)
				}
			}
		}
	}

	if vertexResp.UsageMetadata != nil {
		openAIResp.Usage = convertVertexUsageToImageUsage(vertexResp.UsageMetadata, len(openAIResp.Data))
	}

	return json.Marshal(openAIResp)
}

// convertVertexUsageToImageUsage maps Vertex UsageMetadata to the OpenAI images API usage format.
// The images API uses input_tokens/output_tokens rather than the chat prompt_tokens/completion_tokens.
// Thinking tokens (outside candidatesTokenCount) count as output and are broken out
// as reasoning_tokens, billed at the text rate as on the chat route.
func convertVertexUsageToImageUsage(meta *genai.GenerateContentResponseUsageMetadata, imageCount int) *openai.OpenAIImageUsage {
	inputTokens := int(meta.PromptTokenCount)

	var textTokens, imageTokens int
	for _, detail := range meta.PromptTokensDetails {
		if detail == nil {
			continue
		}
		switch genai.MediaModality(detail.Modality) {
		case genai.MediaModalityImage, genai.MediaModalityVideo:
			imageTokens += int(detail.TokenCount)
		default:
			textTokens += int(detail.TokenCount)
		}
	}
	if textTokens == 0 && imageTokens == 0 {
		textTokens = inputTokens
	}

	outputTokens := int(meta.CandidatesTokenCount)
	outputImageTokens := 0
	for _, detail := range meta.CandidatesTokensDetails {
		if detail == nil {
			continue
		}
		switch genai.MediaModality(detail.Modality) {
		case genai.MediaModalityImage, genai.MediaModalityVideo:
			outputImageTokens += int(detail.TokenCount)
		}
	}
	if outputImageTokens == 0 && imageCount > 0 {
		outputImageTokens = outputTokens
	}
	if outputImageTokens > outputTokens {
		outputImageTokens = outputTokens
	}

	reasoningTokens := max(int(meta.ThoughtsTokenCount), 0)
	outputTokens += reasoningTokens

	var outputTokensDetails *openai.OpenAIImageOutputTokenDetails
	if outputImageTokens > 0 || reasoningTokens > 0 {
		outputTokensDetails = &openai.OpenAIImageOutputTokenDetails{
			ImageTokens:     outputImageTokens,
			ReasoningTokens: reasoningTokens,
		}
	}

	return &openai.OpenAIImageUsage{
		InputTokens: inputTokens,
		InputTokensDetails: openai.OpenAIImageInputTokenDetails{
			TextTokens:  textTokens,
			ImageTokens: imageTokens,
		},
		OutputTokens:        outputTokens,
		OutputTokensDetails: outputTokensDetails,
		TotalTokens:         inputTokens + outputTokens,
	}
}

func parseImageEditFloat(raw, name string) (float64, error) {
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, converterutil.NewInvalidValueError(name)
	}
	return value, nil
}

func readMultipartPartLimit(part *multipart.Part, maxBytes int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(part, maxBytes+1))
	if err != nil {
		return nil, imageValidationError(part.FormName(), "Invalid multipart form data", "invalid_multipart")
	}
	if int64(len(data)) > maxBytes {
		return nil, converterutil.NewRequestEntityTooLargeError(part.FormName(), "Image exceeds the multipart size limit")
	}
	return data, nil
}

func detectImageMIMEType(headerValue string, data []byte) (string, error) {
	mimeType := strings.TrimSpace(headerValue)
	if mimeType == "" || mimeType == "application/octet-stream" {
		mimeType = http.DetectContentType(data)
	}
	if !IsImageMIME(mimeType) {
		return "", fmt.Errorf("unsupported MIME type %q", mimeType)
	}
	return mimeType, nil
}

func clampImageCount(n int) int {
	if n < 1 {
		return 1
	}
	if n > 10 {
		return 10
	}
	return n
}

func imageValidationError(param, message, code string) error {
	return &converterutil.RequestValidationError{Param: param, Message: message, Code: code}
}
