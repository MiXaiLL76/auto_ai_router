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
		return nil, fmt.Errorf("failed to parse OpenAI image request: %w", err)
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
	var imageReq openai.OpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &imageReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI image request: %w", err)
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
			return nil, fmt.Errorf("invalid image generation seed %d", *imageReq.Seed)
		}
		chatReq.Seed = imageReq.Seed
	}
	if imageReq.N != nil && *imageReq.N > 0 {
		n := clampImageCount(*imageReq.N)
		chatReq.N = &n
	}

	return json.Marshal(chatReq)
}

// ImageEditRequestToOpenAIChatRequest converts OpenAI multipart images.edit payload
// to an OpenAI chat request for Gemini image-capable models.
func ImageEditRequestToOpenAIChatRequest(openAIBody []byte, contentType string) ([]byte, error) {
	return imageEditRequestToOpenAIChatRequest(openAIBody, contentType, "")
}

func imageEditRequestToOpenAIChatRequest(openAIBody []byte, contentType, providerModel string) ([]byte, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, fmt.Errorf("failed to parse image edit content type: %w", err)
	}
	if !strings.HasPrefix(mediaType, "multipart/form-data") {
		return nil, fmt.Errorf("image edits require multipart/form-data content type")
	}

	boundary := params["boundary"]
	if boundary == "" {
		return nil, fmt.Errorf("missing multipart boundary in content type")
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
			return nil, fmt.Errorf("failed to read multipart image edit payload: %w", err)
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
			return nil, fmt.Errorf("invalid multipart part %q: %w", formName, mimeErr)
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
		return nil, fmt.Errorf("image edit request missing model field")
	}
	prompt := strings.TrimSpace(fields["prompt"])
	if prompt == "" {
		return nil, fmt.Errorf("image edit request missing prompt field")
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
			return nil, fmt.Errorf("invalid image edit seed %q", rawSeed)
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
	if n := parsePositiveInt(fields["n"]); n > 0 {
		n = clampImageCount(n)
		chatReq.N = &n
	}

	return json.Marshal(chatReq)
}

// VertexChatResponseToOpenAIImage converts Vertex AI chat response with image to OpenAI image format
// Extracts inline image data from chat response and returns it in OpenAI image generation format
func VertexChatResponseToOpenAIImage(vertexBody []byte) ([]byte, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex chat response: %w", err)
	}

	openAIResp := openai.OpenAIImageResponse{
		Created: converterutil.GetCurrentTimestamp(),
		Data:    make([]openai.OpenAIImageData, 0),
	}

	// Extract images from candidates
	for _, candidate := range vertexResp.Candidates {
		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
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
		openAIResp.Usage = convertVertexUsageToImageUsage(vertexResp.UsageMetadata)
	}

	return json.Marshal(openAIResp)
}

// convertVertexUsageToImageUsage maps Vertex UsageMetadata to the OpenAI images API usage format.
// The images API uses input_tokens/output_tokens rather than the chat prompt_tokens/completion_tokens.
func convertVertexUsageToImageUsage(meta *genai.GenerateContentResponseUsageMetadata) *openai.OpenAIImageUsage {
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
	for _, detail := range meta.CandidatesTokensDetails {
		if detail == nil {
			continue
		}
		switch genai.MediaModality(detail.Modality) {
		case genai.MediaModalityImage, genai.MediaModalityVideo:
			// image output tokens counted in CandidatesTokenCount
		}
	}

	return &openai.OpenAIImageUsage{
		InputTokens: inputTokens,
		InputTokensDetails: openai.OpenAIImageInputTokenDetails{
			TextTokens:  textTokens,
			ImageTokens: imageTokens,
		},
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}
}

func parsePositiveInt(raw string) int {
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

func parseImageEditFloat(raw, name string) (float64, error) {
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("invalid image edit %s %q", name, raw)
	}
	return value, nil
}

func readMultipartPartLimit(part *multipart.Part, maxBytes int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(part, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("failed to read multipart part %q: %w", part.FormName(), err)
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("multipart part %q exceeds %d bytes", part.FormName(), maxBytes)
	}
	return data, nil
}

func detectImageMIMEType(headerValue string, data []byte) (string, error) {
	mimeType := strings.TrimSpace(headerValue)
	if mimeType == "" || mimeType == "application/octet-stream" {
		mimeType = http.DetectContentType(data)
	}
	if !strings.HasPrefix(strings.ToLower(mimeType), "image/") {
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
