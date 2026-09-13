package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/models"
)

// imageResponseFacts is what an image response tells us about billing.
type imageResponseFacts struct {
	// Count is the number of delivered images; CountKnown=false means the
	// response carries no recognizable count and the request's "n" is kept.
	Count      int
	CountKnown bool
	// OutputPixels lists width×height per delivered image (0 = unknown size).
	OutputPixels []int64
	// InputImages is the provider-reported number of source images.
	InputImages      int
	InputImagesKnown bool
}

// imageFactsFromResponseBody inspects a non-streaming /v1/images/generations
// or /v1/images/edits response.
//
// The request's "n" is only a hint: some providers ignore it and return a
// single image, others return a batch larger than "n" (sequential/grouped
// generation), and a response that is not an image payload at all delivers
// nothing. The images actually delivered in "data" are what gets billed; a
// provider-reported usage.generated_images is used only when "data" can't be
// read.
func imageFactsFromResponseBody(body []byte) imageResponseFacts {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return imageResponseFacts{CountKnown: true}
	}

	var response struct {
		Data  *[]json.RawMessage `json:"data"`
		Usage *imageUsage        `json:"usage"`
	}
	if err := json.Unmarshal(trimmed, &response); err != nil {
		// Not a JSON object (HTML error page, plain text, bare array...):
		// no image was delivered to the client.
		return imageResponseFacts{CountKnown: true}
	}

	var facts imageResponseFacts
	if delivered, pixels, ok := deliveredImages(response.Data); ok {
		facts.Count, facts.CountKnown = delivered, true
		facts.OutputPixels = pixels
	} else {
		facts.Count, facts.CountKnown = response.Usage.generatedImages()
	}
	facts.InputImages, facts.InputImagesKnown = response.Usage.inputImages()
	return facts
}

// imageFactsFromStreamPayloads reads the most recent usage event among SSE
// data payloads of a streaming image response. Streams only carry totals
// there, so image sizes stay unknown.
func imageFactsFromStreamPayloads(payloads [][]byte) imageResponseFacts {
	for i := len(payloads) - 1; i >= 0; i-- {
		if !bytes.Contains(payloads[i], []byte(`"generated_images"`)) && !bytes.Contains(payloads[i], []byte(`"input_images"`)) {
			continue
		}
		var event struct {
			Usage *imageUsage `json:"usage"`
		}
		if err := json.Unmarshal(payloads[i], &event); err != nil || event.Usage == nil {
			continue
		}
		var facts imageResponseFacts
		facts.Count, facts.CountKnown = event.Usage.generatedImages()
		facts.InputImages, facts.InputImagesKnown = event.Usage.inputImages()
		if facts.CountKnown || facts.InputImagesKnown {
			return facts
		}
	}
	return imageResponseFacts{}
}

type imageUsage struct {
	GeneratedImages *int `json:"generated_images"`
	InputImages     *int `json:"input_images"`
}

func (u *imageUsage) generatedImages() (int, bool) {
	if u == nil || u.GeneratedImages == nil || *u.GeneratedImages < 0 {
		return 0, false
	}
	return *u.GeneratedImages, true
}

func (u *imageUsage) inputImages() (int, bool) {
	if u == nil || u.InputImages == nil || *u.InputImages < 0 {
		return 0, false
	}
	return *u.InputImages, true
}

// deliveredImages counts the entries of a response "data" array that carry an
// image and records each one's pixel count from its "size" ("WIDTHxHEIGHT").
// ok=false when data is absent or holds an entry shape we don't recognize.
func deliveredImages(data *[]json.RawMessage) (delivered int, pixels []int64, ok bool) {
	if data == nil {
		return 0, nil, false
	}
	for _, raw := range *data {
		var item struct {
			URL     string          `json:"url"`
			B64JSON string          `json:"b64_json"`
			Size    string          `json:"size"`
			Error   json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			return 0, nil, false
		}
		switch {
		case item.URL != "" || item.B64JSON != "":
			delivered++
			pixels = append(pixels, imagePixels(item.Size))
		case len(item.Error) > 0 && !bytes.Equal(item.Error, []byte("null")):
			// A per-image failure (e.g. filtered output) is not a delivered image.
		default:
			// An entry shape we don't recognize — don't guess.
			return 0, nil, false
		}
	}
	return delivered, pixels, true
}

// maxImagePixels caps parsed dimensions so a bogus size can't overflow.
const maxImagePixels = 1 << 40

// imagePixels parses "WIDTHxHEIGHT" into width×height, or 0 when it isn't one.
func imagePixels(size string) int64 {
	width, height, found := strings.Cut(strings.ToLower(strings.TrimSpace(size)), "x")
	if !found {
		return 0
	}
	w, errW := strconv.ParseInt(width, 10, 64)
	h, errH := strconv.ParseInt(height, 10, 64)
	if errW != nil || errH != nil || w <= 0 || h <= 0 || w > maxImagePixels/h {
		return 0
	}
	return w * h
}

// imageParamsIgnoredForPricing are request fields that never select a price
// tier; they are skipped so free text doesn't end up in billing state.
var imageParamsIgnoredForPricing = map[string]struct{}{
	"model": {}, "prompt": {}, "negative_prompt": {}, "user": {},
}

// maxImageParamRawLen bounds the raw JSON of a parameter worth decoding:
// anything longer can't be a tier-selecting value (see
// models.NormalizeImageParamValue), and skipping it avoids decoding inline
// base64 images.
const maxImageParamRawLen = 256

// imageBillingRequestFromBody extracts the request-side facts per-image price
// tiers need: the operation, short scalar parameters (exact names, canonical
// values) and, for edits, how many source images were sent. Generation
// requests report source images through the response usage instead (a
// generation endpoint may ignore an "image" field), so they are not counted
// from the request.
func imageBillingRequestFromBody(body []byte, contentType string, edit bool) *converter.ImageBillingDetails {
	details := &converter.ImageBillingDetails{
		Operation:     converter.ImageOperationGeneration,
		RequestParams: map[string]string{},
	}
	if edit {
		details.Operation = converter.ImageOperationEdit
	}

	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		collectMultipartImageRequest(body, contentType, edit, details)
		return details
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return details
	}
	for name, raw := range fields {
		if edit && (name == "image" || name == "images") {
			details.InputImages += jsonImageCount(raw)
			continue
		}
		if _, skip := imageParamsIgnoredForPricing[name]; skip || len(raw) > maxImageParamRawLen {
			continue
		}
		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			continue
		}
		if normalized, ok := models.NormalizeImageParamValue(value); ok {
			details.RequestParams[name] = normalized
		}
	}
	return details
}

// jsonImageCount counts source images in an "image"/"images" JSON value: a
// single non-empty string or object is one image, an array holds one per
// non-empty string or object entry. Values are classified by their raw JSON
// so inline base64 images are never decoded.
func jsonImageCount(raw json.RawMessage) int {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return 0
	}
	switch trimmed[0] {
	case '"', '{':
		return jsonImageEntryCount(trimmed)
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return 0
		}
		count := 0
		for _, item := range items {
			count += jsonImageEntryCount(bytes.TrimSpace(item))
		}
		return count
	default:
		return 0
	}
}

func jsonImageEntryCount(item []byte) int {
	switch {
	case len(item) >= 2 && item[0] == '"':
		if len(bytes.TrimSpace(item[1:len(item)-1])) > 0 {
			return 1
		}
	case len(item) >= 2 && item[0] == '{':
		return 1
	}
	return 0
}

// collectMultipartImageRequest fills details from a multipart image request:
// short text fields become parameters, file parts named image/images count as
// source images for edits.
func collectMultipartImageRequest(body []byte, contentType string, edit bool, details *converter.ImageBillingDetails) {
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil || params["boundary"] == "" {
		return
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	for {
		part, err := reader.NextPart()
		if err != nil {
			return
		}
		name := strings.TrimSuffix(part.FormName(), "[]")
		if part.FileName() != "" {
			if edit && (name == "image" || name == "images") {
				details.InputImages++
			}
			continue
		}
		if _, skip := imageParamsIgnoredForPricing[name]; skip || name == "" {
			continue
		}
		data, err := io.ReadAll(io.LimitReader(part, maxImageParamRawLen+1))
		if err != nil {
			return
		}
		if len(data) > maxImageParamRawLen {
			continue
		}
		if normalized, ok := models.NormalizeImageFormValue(string(data)); ok {
			details.RequestParams[name] = normalized
		}
	}
}

// imageBillingDetails returns a fresh copy of the request-side facts so
// response facts can be layered on without mutating shared request state.
func (l *RequestLogContext) imageBillingDetails() *converter.ImageBillingDetails {
	if l.imageBillingRequest == nil {
		return nil
	}
	details := *l.imageBillingRequest
	details.OutputPixels = nil
	return &details
}

// setImageCountFromBody records the billable image count and per-image pricing
// facts for a successful non-streaming image response. A count read from the
// body is authoritative and is not replaced by the request-derived default in
// logSpendToLiteLLMDB; when the body carries no count, the request's "n" is
// used as before.
func (l *RequestLogContext) setImageCountFromBody(body []byte) {
	if l.TokenUsage == nil {
		l.TokenUsage = &converter.TokenUsage{}
	}
	facts := imageFactsFromResponseBody(body)
	if facts.CountKnown {
		l.TokenUsage.ImageCount = facts.Count
		l.ImageCountReported = true
	} else {
		l.TokenUsage.ImageCount = l.ImageCount
	}
	l.applyImageResponseFacts(facts)
}

// setImageCountFromStreamChunk is the streaming counterpart for the last data
// chunk of a completed stream: it only overrides the request-derived count
// when the terminal usage event reports one.
func (l *RequestLogContext) setImageCountFromStreamChunk(chunk []byte) {
	l.applyImageStreamFacts(imageFactsFromStreamPayloads(splitSSEPayloads(chunk, nil)))
}

// observeImageStreamPayloads remembers the latest image usage seen while a
// stream is relayed chunk by chunk; logSpendToLiteLLMDB applies it once the
// request is known to have succeeded.
func (l *RequestLogContext) observeImageStreamPayloads(payloads [][]byte) {
	if facts := imageFactsFromStreamPayloads(payloads); facts.CountKnown || facts.InputImagesKnown {
		l.imageStreamFacts = &facts
	}
}

func (l *RequestLogContext) applyImageStreamFacts(facts imageResponseFacts) {
	if l.TokenUsage == nil {
		l.TokenUsage = &converter.TokenUsage{}
	}
	if facts.CountKnown {
		l.TokenUsage.ImageCount = facts.Count
		l.ImageCountReported = true
	}
	l.applyImageResponseFacts(facts)
}

func (l *RequestLogContext) applyImageResponseFacts(facts imageResponseFacts) {
	details := l.imageBillingDetails()
	if details == nil {
		return
	}
	details.OutputPixels = facts.OutputPixels
	if facts.InputImagesKnown {
		details.InputImages = facts.InputImages
	}
	l.TokenUsage.ImageBilling = details
}
