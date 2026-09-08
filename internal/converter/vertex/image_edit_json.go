package vertex

import (
	"encoding/json"
	"net/url"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

func imageEditJSONToOpenAIChatRequest(body []byte, model string) ([]byte, error) {
	chatBody, err := imageRequestToOpenAIChatRequest(body, model)
	if err != nil {
		return nil, err
	}
	var fields struct {
		Image  json.RawMessage `json:"image"`
		Images json.RawMessage `json:"images"`
		Mask   json.RawMessage `json:"mask"`
	}
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, imageJSONValidationError(err)
	}
	if len(fields.Image) > 0 && len(fields.Images) > 0 {
		return nil, imageValidationError("image", "Invalid parameter value", "invalid_value")
	}
	rawImages := fields.Image
	if len(fields.Images) > 0 {
		rawImages = fields.Images
	}
	var images []json.RawMessage
	if len(rawImages) > 0 && rawImages[0] == '[' {
		if err := json.Unmarshal(rawImages, &images); err != nil {
			return nil, imageJSONValidationError(err)
		}
	} else if len(rawImages) > 0 && string(rawImages) != "null" {
		images = append(images, rawImages)
	}
	if len(images) == 0 {
		return nil, imageValidationError("image", "Missing required parameter", "missing_required_parameter")
	}
	var chat openai.OpenAIRequest
	if err := json.Unmarshal(chatBody, &chat); err != nil {
		return nil, err
	}
	blocks := []openai.ContentBlock{{Type: "text", Text: chat.Messages[0].Content.(string)}}
	for _, raw := range images {
		imageURL, err := imageEditJSONURL(raw, "image")
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, openai.ContentBlock{Type: "image_url", ImageURL: imageURL})
	}
	if len(fields.Mask) > 0 && string(fields.Mask) != "null" {
		mask, err := imageEditJSONURL(fields.Mask, "mask")
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, openai.ContentBlock{Type: "text", Text: "Use the provided mask image to constrain the edit."}, openai.ContentBlock{Type: "image_url", ImageURL: mask})
	}
	chat.Messages[0].Content = blocks
	return json.Marshal(chat)
}

func imageEditJSONURL(raw json.RawMessage, param string) (*openai.ImageURL, error) {
	var value string
	if json.Unmarshal(raw, &value) != nil {
		var object struct {
			URL      string `json:"url"`
			ImageURL string `json:"image_url"`
		}
		if json.Unmarshal(raw, &object) != nil {
			return nil, imageValidationError(param, "Invalid parameter type", "invalid_type")
		}
		value = object.ImageURL
		if value == "" {
			value = object.URL
		}
	}
	if strings.HasPrefix(value, "data:") {
		part, err := parseDataURLToPart(value)
		if err != nil {
			return nil, err
		}
		if part == nil || part.InlineData == nil || !strings.HasPrefix(part.InlineData.MIMEType, "image/") || len(part.InlineData.Data) == 0 {
			return nil, imageValidationError(param, "Invalid image data", "invalid_image")
		}
	} else {
		parsed, err := url.Parse(value)
		if err != nil || parsed.Host == "" || parsed.Scheme != "https" && parsed.Scheme != "http" && parsed.Scheme != "gs" || parseURLToPart(value, nil) == nil {
			return nil, imageValidationError(param, "Invalid image data", "invalid_image")
		}
	}
	return &openai.ImageURL{URL: value}, nil
}
