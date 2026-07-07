package sosana

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

const maxMultipartImageBytes = 20 * 1024 * 1024

const (
	StatusProcessing = "PROCESSING"
	StatusCompleted  = "COMPLETED"
	StatusFailed     = "FAILED"
	StatusModerated  = "MODERATED"
)

type BananaCreateRequest struct {
	Prompt             string   `json:"prompt"`
	ImageURLs          []string `json:"image_urls,omitempty"`
	Model              string   `json:"model,omitempty"`
	AspectRatio        string   `json:"aspect_ratio,omitempty"`
	ImageSize          string   `json:"image_size,omitempty"`
	PromptOptimization bool     `json:"prompt_optimization"`
}

type BananaTaskResponse struct {
	UID             string  `json:"uid"`
	Status          string  `json:"status"`
	Prompt          string  `json:"prompt"`
	CreatedAt       string  `json:"created_at"`
	OptimizedPrompt string  `json:"optimized_prompt"`
	ResultFileURL   *string `json:"result_file_url"`
	Error           *string `json:"error"`
}

type openAIImageRequest struct {
	openai.OpenAIImageRequest
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Ratio       string `json:"ratio,omitempty"`
	ImageSize   string `json:"image_size,omitempty"`
}

func ImageGenerationRequest(openAIBody []byte, modelID string) ([]byte, string, error) {
	if reason := UnsupportedRequest("/v1/images/generations", openAIBody, "application/json"); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}

	var req openAIImageRequest
	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, "", fmt.Errorf("failed to parse OpenAI image request: %w", err)
	}
	if err := validateImageCount(req.N); err != nil {
		return nil, "", err
	}
	if err := validateResponseFormat(req.ResponseFormat); err != nil {
		return nil, "", err
	}
	if err := validateOutputFormat(req.OutputFormat); err != nil {
		return nil, "", err
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return nil, "", fmt.Errorf("image generation request missing prompt")
	}
	imageSize, err := imageSize(req.ImageSize, req.Size)
	if err != nil {
		return nil, "", err
	}
	concreteModel := providerModel(modelID, req.Model, imageSize)
	if reason := UnsupportedModel(concreteModel); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}
	body, err := json.Marshal(BananaCreateRequest{
		Prompt:             prompt,
		Model:              concreteModel,
		AspectRatio:        aspectRatio(req.AspectRatio, req.Ratio, req.Size),
		ImageSize:          imageSize,
		PromptOptimization: false,
	})
	if err != nil {
		return nil, "", err
	}
	return body, concreteModel, nil
}

func ImageEditRequest(openAIBody []byte, contentType string, modelID string) ([]byte, string, error) {
	if reason := UnsupportedRequest("/v1/images/edits", openAIBody, contentType); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}

	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, "", fmt.Errorf("failed to parse image edit content type: %w", err)
	}
	if !strings.HasPrefix(mediaType, "multipart/form-data") {
		return nil, "", fmt.Errorf("image edits require multipart/form-data content type")
	}
	boundary := params["boundary"]
	if boundary == "" {
		return nil, "", fmt.Errorf("missing multipart boundary in content type")
	}

	fields := make(map[string]string)
	imageURLs := make([]string, 0, 1)
	reader := multipart.NewReader(bytes.NewReader(openAIBody), boundary)
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", fmt.Errorf("failed to read multipart image edit payload: %w", err)
		}

		formName := part.FormName()
		if formName == "" {
			continue
		}
		data, err := readLimited(part, maxMultipartImageBytes)
		if err != nil {
			return nil, "", err
		}
		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}
		if formName == "mask" {
			return nil, "", fmt.Errorf("image edits do not support mask")
		}
		if formName != "image" && formName != "images" && formName != "image[]" {
			continue
		}
		mimeType := detectImageMIMEType(part.Header.Get("Content-Type"), data)
		if mimeType != "image/png" {
			return nil, "", fmt.Errorf("image edits support PNG images only")
		}
		imageURLs = append(imageURLs, "data:"+mimeType+";base64,"+base64.StdEncoding.EncodeToString(data))
	}

	if len(imageURLs) > maxInputImages {
		return nil, "", fmt.Errorf("image edits support up to %d input images", maxInputImages)
	}
	if err := validateImageCountString(fields["n"]); err != nil {
		return nil, "", err
	}
	if err := validateResponseFormat(fields["response_format"]); err != nil {
		return nil, "", err
	}
	if err := validateOutputFormat(fields["output_format"]); err != nil {
		return nil, "", err
	}
	prompt := strings.TrimSpace(fields["prompt"])
	if prompt == "" {
		return nil, "", fmt.Errorf("image edit request missing prompt field")
	}
	if len(imageURLs) == 0 {
		return nil, "", fmt.Errorf("image edit request missing image")
	}
	imageSize, err := imageSize(fields["image_size"], fields["size"])
	if err != nil {
		return nil, "", err
	}
	concreteModel := providerModel(modelID, fields["model"], imageSize)
	if reason := UnsupportedModel(concreteModel); reason != "" {
		return nil, "", fmt.Errorf("%s", reason)
	}
	body, err := json.Marshal(BananaCreateRequest{
		Prompt:             prompt,
		ImageURLs:          imageURLs,
		Model:              concreteModel,
		AspectRatio:        aspectRatio(fields["aspect_ratio"], fields["ratio"], fields["size"]),
		ImageSize:          imageSize,
		PromptOptimization: false,
	})
	if err != nil {
		return nil, "", err
	}
	return body, concreteModel, nil
}

func OpenAIImageResponse(task BananaTaskResponse, image []byte) ([]byte, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("image task completed without image bytes")
	}
	resp := openai.OpenAIImageResponse{
		Created: createdAtUnix(task.CreatedAt),
		Data: []openai.OpenAIImageData{
			{
				B64JSON:       base64.StdEncoding.EncodeToString(image),
				RevisedPrompt: strings.TrimSpace(task.OptimizedPrompt),
			},
		},
	}
	return json.Marshal(resp)
}

func providerModel(modelID, requestModel, imageSize string) string {
	model := strings.TrimSpace(modelID)
	if model == "" {
		model = strings.TrimSpace(requestModel)
	}
	if model == "" {
		return ""
	}
	if strings.EqualFold(model, "google/gemini-3.1-flash-image-preview") {
		return "banana-2-" + strings.ToLower(imageSize) + "-compliant"
	}
	return strings.ReplaceAll(model, "{image_size}", strings.ToLower(imageSize))
}

func createdAtUnix(value string) int64 {
	if value == "" {
		return converterutil.GetCurrentTimestamp()
	}
	if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return ts.Unix()
	}
	return converterutil.GetCurrentTimestamp()
}

func CreateURL(baseURL string) string {
	return strings.TrimSuffix(baseURL, "/") + "/api/banana/create-async"
}

func PollURL(baseURL, uid string) string {
	return strings.TrimSuffix(baseURL, "/") + "/api/banana/" + uid
}

func SizeToAspectRatio(size string) string {
	switch strings.TrimSpace(size) {
	case "", "auto":
		return "auto"
	case "256x256", "512x512", "1024x1024", "2048x2048", "4096x4096":
		return "1:1"
	case "1024x1536", "2048x3072":
		return "2:3"
	case "1536x1024", "3072x2048":
		return "3:2"
	case "1024x1792", "1080x1920", "2048x3584", "4096x7168":
		return "9:16"
	case "1792x1024", "1920x1080", "3584x2048", "7168x4096":
		return "16:9"
	case "1024x768", "2048x1536", "4096x3072":
		return "4:3"
	case "768x1024", "1536x2048", "3072x4096":
		return "3:4"
	case "1024x819", "2048x1638", "4096x3276":
		return "5:4"
	case "819x1024", "1638x2048", "3276x4096":
		return "4:5"
	case "2016x864", "4032x1728":
		return "21:9"
	default:
		return "auto"
	}
}

func aspectRatio(explicit, ratio, size string) string {
	if value := strings.TrimSpace(explicit); value != "" {
		return value
	}
	if value := strings.TrimSpace(ratio); value != "" {
		return value
	}
	return SizeToAspectRatio(size)
}

func imageSize(explicit, size string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		if value, ok := normalizeImageSize(explicit); ok {
			return value, nil
		}
		return "", fmt.Errorf("image_size is unsupported")
	}
	if value, ok := imageSizeFromExactSize(size); ok {
		return value, nil
	}
	return "", fmt.Errorf("size is unsupported")
}

func normalizeImageSize(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "auto", "1k":
		return "1K", true
	case "2k":
		return "2K", true
	case "4k":
		return "4K", true
	default:
		return "", false
	}
}

func imageSizeFromExactSize(size string) (string, bool) {
	switch strings.TrimSpace(size) {
	case "", "auto",
		"256x256", "512x512", "1024x1024", "1024x1536", "1536x1024",
		"1024x1792", "1792x1024", "1080x1920", "1920x1080",
		"1024x768", "768x1024", "1024x819", "819x1024", "2016x864":
		return "1K", true
	case "2048x2048", "2048x3072", "3072x2048", "2048x3584", "3584x2048",
		"2048x1536", "1536x2048", "2048x1638", "1638x2048", "4032x1728":
		return "2K", true
	case "4096x4096", "4096x7168", "7168x4096", "4096x3072", "3072x4096", "4096x3276", "3276x4096":
		return "4K", true
	default:
		return "", false
	}
}

func validateImageCount(n *int) error {
	if n == nil || *n == 1 {
		return nil
	}
	return fmt.Errorf("image requests support n=1 only")
}

func validateResponseFormat(format string) error {
	if strings.EqualFold(strings.TrimSpace(format), "url") {
		return fmt.Errorf("response_format=url is unsupported for this image model")
	}
	return nil
}

func validateImageCountString(raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fmt.Errorf("invalid image count: %w", err)
	}
	if n != 1 {
		return fmt.Errorf("image requests support n=1 only")
	}
	return nil
}

func readLimited(r io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, fmt.Errorf("failed to read multipart part: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("multipart image exceeds %d bytes", limit)
	}
	return data, nil
}

func detectImageMIMEType(header string, data []byte) string {
	header = strings.ToLower(strings.TrimSpace(header))
	if strings.HasPrefix(header, "image/") {
		mediaType, _, err := mime.ParseMediaType(header)
		if err == nil {
			return mediaType
		}
		return strings.TrimSpace(strings.Split(header, ";")[0])
	}
	detected := http.DetectContentType(data)
	if strings.HasPrefix(detected, "image/") {
		return detected
	}
	return "application/octet-stream"
}
