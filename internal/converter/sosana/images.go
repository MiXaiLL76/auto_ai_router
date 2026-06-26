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
	Prompt      string   `json:"prompt"`
	ImageURLs   []string `json:"image_urls,omitempty"`
	Model       string   `json:"model,omitempty"`
	AspectRatio string   `json:"aspect_ratio,omitempty"`
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

func ImageGenerationRequest(openAIBody []byte, modelID string) ([]byte, error) {
	var req openai.OpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI image request: %w", err)
	}
	if err := validateImageCount(req.N); err != nil {
		return nil, err
	}
	if err := validateResponseFormat(req.ResponseFormat); err != nil {
		return nil, err
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return nil, fmt.Errorf("image generation request missing prompt")
	}
	return json.Marshal(BananaCreateRequest{
		Prompt:      prompt,
		Model:       providerModel(modelID, req.Model),
		AspectRatio: SizeToAspectRatio(req.Size),
	})
}

func ImageEditRequest(openAIBody []byte, contentType string, modelID string) ([]byte, error) {
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

	fields := make(map[string]string)
	imageURLs := make([]string, 0, 1)
	reader := multipart.NewReader(bytes.NewReader(openAIBody), boundary)
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
		data, err := readLimited(part, maxMultipartImageBytes)
		if err != nil {
			return nil, err
		}
		if part.FileName() == "" {
			fields[formName] = strings.TrimSpace(string(data))
			continue
		}
		if formName == "mask" {
			return nil, fmt.Errorf("sosana image edits do not support mask")
		}
		if formName != "image" && formName != "images" && formName != "image[]" {
			continue
		}
		mimeType := detectImageMIMEType(part.Header.Get("Content-Type"), data)
		imageURLs = append(imageURLs, "data:"+mimeType+";base64,"+base64.StdEncoding.EncodeToString(data))
	}

	if err := validateImageCountString(fields["n"]); err != nil {
		return nil, err
	}
	if err := validateResponseFormat(fields["response_format"]); err != nil {
		return nil, err
	}
	prompt := strings.TrimSpace(fields["prompt"])
	if prompt == "" {
		return nil, fmt.Errorf("image edit request missing prompt field")
	}
	if len(imageURLs) == 0 {
		return nil, fmt.Errorf("image edit request missing image")
	}
	return json.Marshal(BananaCreateRequest{
		Prompt:      prompt,
		ImageURLs:   imageURLs,
		Model:       providerModel(modelID, fields["model"]),
		AspectRatio: SizeToAspectRatio(fields["size"]),
	})
}

func OpenAIImageResponse(task BananaTaskResponse, image []byte) ([]byte, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("sosana task completed without image bytes")
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

func providerModel(modelID, requestModel string) string {
	if model := strings.TrimSpace(modelID); model != "" {
		return model
	}
	return strings.TrimSpace(requestModel)
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
	case "256x256", "512x512", "1024x1024", "4096x4096":
		return "1:1"
	case "1024x1536":
		return "2:3"
	case "1536x1024":
		return "3:2"
	case "1024x1792", "1080x1920", "4096x7168":
		return "9:16"
	case "1792x1024", "1920x1080", "7168x4096":
		return "16:9"
	default:
		return "auto"
	}
}

func validateImageCount(n *int) error {
	if n == nil || *n == 1 {
		return nil
	}
	return fmt.Errorf("sosana supports n=1 only")
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
		return fmt.Errorf("sosana supports n=1 only")
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
		return header
	}
	detected := http.DetectContentType(data)
	if strings.HasPrefix(detected, "image/") {
		return detected
	}
	return "application/octet-stream"
}
