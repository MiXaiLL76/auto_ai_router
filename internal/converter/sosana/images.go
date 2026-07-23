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
	if spec, ok := imageSpecFromExactSize(size); ok {
		return spec.aspectRatio
	}
	return "auto"
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
	if spec, ok := imageSpecFromExactSize(size); ok {
		return spec.imageSize, true
	}
	return "", false
}

type exactImageSpec struct {
	imageSize   string
	aspectRatio string
}

func imageSpecFromExactSize(size string) (exactImageSpec, bool) {
	switch strings.TrimSpace(size) {
	case "", "auto":
		return exactImageSpec{imageSize: "1K", aspectRatio: "auto"}, true

	case "1024x1024":
		return exactImageSpec{imageSize: "1K", aspectRatio: "1:1"}, true
	case "512x2048":
		return exactImageSpec{imageSize: "1K", aspectRatio: "1:4"}, true
	case "384x3072":
		return exactImageSpec{imageSize: "1K", aspectRatio: "1:8"}, true
	case "848x1264":
		return exactImageSpec{imageSize: "1K", aspectRatio: "2:3"}, true
	case "1264x848":
		return exactImageSpec{imageSize: "1K", aspectRatio: "3:2"}, true
	case "896x1200":
		return exactImageSpec{imageSize: "1K", aspectRatio: "3:4"}, true
	case "2048x512":
		return exactImageSpec{imageSize: "1K", aspectRatio: "4:1"}, true
	case "1200x896":
		return exactImageSpec{imageSize: "1K", aspectRatio: "4:3"}, true
	case "928x1152":
		return exactImageSpec{imageSize: "1K", aspectRatio: "4:5"}, true
	case "1152x928":
		return exactImageSpec{imageSize: "1K", aspectRatio: "5:4"}, true
	case "3072x384":
		return exactImageSpec{imageSize: "1K", aspectRatio: "8:1"}, true
	case "768x1376":
		return exactImageSpec{imageSize: "1K", aspectRatio: "9:16"}, true
	case "1376x768":
		return exactImageSpec{imageSize: "1K", aspectRatio: "16:9"}, true
	case "1584x672":
		return exactImageSpec{imageSize: "1K", aspectRatio: "21:9"}, true

	case "2048x2048":
		return exactImageSpec{imageSize: "2K", aspectRatio: "1:1"}, true
	case "1024x4096":
		return exactImageSpec{imageSize: "2K", aspectRatio: "1:4"}, true
	case "768x6144":
		return exactImageSpec{imageSize: "2K", aspectRatio: "1:8"}, true
	case "1696x2528":
		return exactImageSpec{imageSize: "2K", aspectRatio: "2:3"}, true
	case "2528x1696":
		return exactImageSpec{imageSize: "2K", aspectRatio: "3:2"}, true
	case "1792x2400":
		return exactImageSpec{imageSize: "2K", aspectRatio: "3:4"}, true
	case "4096x1024":
		return exactImageSpec{imageSize: "2K", aspectRatio: "4:1"}, true
	case "2400x1792":
		return exactImageSpec{imageSize: "2K", aspectRatio: "4:3"}, true
	case "1856x2304":
		return exactImageSpec{imageSize: "2K", aspectRatio: "4:5"}, true
	case "2304x1856":
		return exactImageSpec{imageSize: "2K", aspectRatio: "5:4"}, true
	case "6144x768":
		return exactImageSpec{imageSize: "2K", aspectRatio: "8:1"}, true
	case "1536x2752":
		return exactImageSpec{imageSize: "2K", aspectRatio: "9:16"}, true
	case "2752x1536":
		return exactImageSpec{imageSize: "2K", aspectRatio: "16:9"}, true
	case "3168x1344":
		return exactImageSpec{imageSize: "2K", aspectRatio: "21:9"}, true

	case "4096x4096":
		return exactImageSpec{imageSize: "4K", aspectRatio: "1:1"}, true
	case "2048x8192":
		return exactImageSpec{imageSize: "4K", aspectRatio: "1:4"}, true
	case "1536x12288":
		return exactImageSpec{imageSize: "4K", aspectRatio: "1:8"}, true
	case "3392x5056":
		return exactImageSpec{imageSize: "4K", aspectRatio: "2:3"}, true
	case "5056x3392":
		return exactImageSpec{imageSize: "4K", aspectRatio: "3:2"}, true
	case "3584x4800":
		return exactImageSpec{imageSize: "4K", aspectRatio: "3:4"}, true
	case "8192x2048":
		return exactImageSpec{imageSize: "4K", aspectRatio: "4:1"}, true
	case "4800x3584":
		return exactImageSpec{imageSize: "4K", aspectRatio: "4:3"}, true
	case "3712x4608":
		return exactImageSpec{imageSize: "4K", aspectRatio: "4:5"}, true
	case "4608x3712":
		return exactImageSpec{imageSize: "4K", aspectRatio: "5:4"}, true
	case "12288x1536":
		return exactImageSpec{imageSize: "4K", aspectRatio: "8:1"}, true
	case "3072x5504":
		return exactImageSpec{imageSize: "4K", aspectRatio: "9:16"}, true
	case "5504x3072":
		return exactImageSpec{imageSize: "4K", aspectRatio: "16:9"}, true
	case "6336x2688":
		return exactImageSpec{imageSize: "4K", aspectRatio: "21:9"}, true
	default:
		return exactImageSpec{}, false
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
