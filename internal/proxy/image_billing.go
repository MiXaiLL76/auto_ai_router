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
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/models"
)

// Image generation/edit requests are billed per image, in three steps shared
// by every proxy path:
//
//  1. imageRequestFromBody reads the request once as it arrives: the number of
//     images asked for and the inputs per-image price tiers depend on.
//  2. observeImageResponseBody (non-streaming) and observeImageStreamPayloads
//     (every streaming relay) record what the provider response reported.
//  3. finalizeImageUsage, run by logSpendToLiteLLMDB for a successful request,
//     turns both into TokenUsage.ImageCount and TokenUsage.ImageBilling.
//
// Response facts stay on the log context until step 3, so paths that replace
// TokenUsage wholesale (retries, fallbacks, stream usage) can't drop them.

// imageResponseFacts is what an image response tells us about billing.
type imageResponseFacts struct {
	// Count is the number of images to bill; CountKnown=false means the
	// response says nothing about it and the request's "n" is billed.
	Count      int
	CountKnown bool
	// OutputPixels lists width×height of the delivered images the response
	// recognizably carries, in response order (0 = unknown size); images
	// beyond it are of unknown size.
	OutputPixels []int64
	// InputImages is the provider-reported number of source images.
	InputImages      int
	InputImagesKnown bool
}

// imageFactsFromResponseBody inspects a non-streaming /v1/images/generations
// or /v1/images/edits response to a request that asked for requested images.
//
// The request's "n" is only a hint: some providers ignore it and return a
// single image, others return a batch larger than "n" (sequential/grouped
// generation). The images in "data" are what gets billed: entries carrying an
// image always count, failed and empty entries never do, and entries of a
// shape we don't recognize may or may not be images — the provider's
// usage.generated_images (or, failing that, "n") decides how many of them
// count. A provider-reported count alone is used when "data" is absent.
//
// A body that isn't a JSON object (empty, an HTML error page, raw image
// bytes) tells nothing about what the client received, so the count stays
// unknown rather than zero.
func imageFactsFromResponseBody(body []byte, requested int) imageResponseFacts {
	var response struct {
		Data  *[]json.RawMessage `json:"data"`
		Usage json.RawMessage    `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		// Decoding "data" straight into entries avoids copying a payload that
		// can hold megabytes of base64, but fails the whole body when "data"
		// isn't an array; the usage can still be read then.
		var usageOnly struct {
			Usage json.RawMessage `json:"usage"`
		}
		if json.Unmarshal(body, &usageOnly) != nil {
			return imageResponseFacts{}
		}
		response.Data, response.Usage = nil, usageOnly.Usage
	}

	var facts imageResponseFacts
	usage := decodeImageUsage(response.Usage)
	facts.InputImages, facts.InputImagesKnown = usage.inputImages()
	reported, reportedKnown := usage.generatedImages()
	if entries, ok := classifyImageEntries(response.Data); ok {
		hint := requested
		if reportedKnown {
			hint = reported
		}
		facts.Count = min(max(hint, entries.images), entries.images+entries.unrecognized)
		facts.CountKnown = true
		facts.OutputPixels = entries.pixels
	} else if reportedKnown {
		facts.Count, facts.CountKnown = reported, true
	}
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
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(payloads[i], &event); err != nil {
			continue
		}
		usage := decodeImageUsage(event.Usage)
		var facts imageResponseFacts
		facts.Count, facts.CountKnown = usage.generatedImages()
		facts.InputImages, facts.InputImagesKnown = usage.inputImages()
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

// decodeImageUsage reads the image counters of a usage object, or nil when
// there is none or it doesn't have the expected shape.
func decodeImageUsage(raw json.RawMessage) *imageUsage {
	var usage imageUsage
	if len(raw) == 0 || json.Unmarshal(raw, &usage) != nil {
		return nil
	}
	return &usage
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

// imageEntries classifies the entries of a response "data" array.
type imageEntries struct {
	// images counts entries carrying an image ("url" or "b64_json").
	images int
	// unrecognized counts entries that are neither an image, a per-image
	// failure ("error") nor empty.
	unrecognized int
	// pixels lists width×height of each image entry from its "size"
	// ("WIDTHxHEIGHT"), 0 when absent or unparsable.
	pixels []int64
}

// classifyImageEntries sorts a response "data" array into image entries and
// entries of unknown shape. ok=false when data is absent (or null).
func classifyImageEntries(data *[]json.RawMessage) (entries imageEntries, ok bool) {
	if data == nil || *data == nil {
		return imageEntries{}, false
	}
	for _, raw := range *data {
		var item struct {
			URL     json.RawMessage `json:"url"`
			B64JSON json.RawMessage `json:"b64_json"`
			Size    json.RawMessage `json:"size"`
			Error   json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			// Not an object (a bare string, a number...).
			entries.unrecognized++
			continue
		}
		switch {
		case isNonEmptyJSONString(item.URL) || isNonEmptyJSONString(item.B64JSON):
			entries.images++
			var size string
			_ = json.Unmarshal(item.Size, &size)
			entries.pixels = append(entries.pixels, imagePixels(size))
		case !isEmptyJSONValue(item.Error):
			// A per-image failure (e.g. filtered output) is not a delivered image.
		case isEmptyImageEntry(raw):
			// null, {} or {"url":null}: nothing was delivered.
		default:
			entries.unrecognized++
		}
	}
	return entries, true
}

// isNonEmptyJSONString reports whether raw is a JSON string with content,
// without decoding it (b64_json values can be megabytes long).
func isNonEmptyJSONString(raw json.RawMessage) bool {
	return len(raw) > 2 && raw[0] == '"'
}

// isEmptyJSONValue reports whether raw is absent, null or "".
func isEmptyJSONValue(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || bytes.Equal(trimmed, []byte(`""`))
}

// isEmptyImageEntry reports whether a data entry carries no value at all.
// Only entries without an image or error get here, so they are small.
func isEmptyImageEntry(raw json.RawMessage) bool {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return false
	}
	for _, value := range fields {
		if !isEmptyJSONValue(value) {
			return false
		}
	}
	return true
}

// maxImagePixels caps parsed dimensions so a bogus size can't overflow.
const maxImagePixels = 1 << 40

// imagePixels parses "WIDTHxHEIGHT" into width×height, or 0 when it isn't one.
func imagePixels(size string) int64 {
	width, height, ratio, ok := converterutil.ParseImageDimensions(size)
	if !ok || ratio || int64(width) > maxImagePixels/int64(height) {
		return 0
	}
	return int64(width) * int64(height)
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

// imageRequestFromBody reads an image generation/edit request in one pass.
//
// requested is the number of images it asks for: a positive "n", otherwise 1.
// details are the request-side facts per-image price tiers need: the
// operation, short scalar parameters (exact names, canonical values) and, for
// edits, how many source images were sent. Generation requests report source
// images through the response usage instead (a generation endpoint may ignore
// an "image" field), so they are not counted from the request.
func imageRequestFromBody(body []byte, contentType string, edit bool) (requested int, details *converter.ImageBillingDetails) {
	requested = 1
	details = &converter.ImageBillingDetails{
		Operation:     converter.ImageOperationGeneration,
		RequestParams: map[string]string{},
	}
	if edit {
		details.Operation = converter.ImageOperationEdit
	}

	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		if n := collectMultipartImageRequest(body, contentType, edit, details); n > 0 {
			requested = n
		}
		return requested, details
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return requested, details
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
		if name == "n" {
			var n int
			if json.Unmarshal(raw, &n) == nil && n > 0 {
				requested = n
			}
		}
		if normalized, ok := models.NormalizeImageParamValue(value); ok {
			details.RequestParams[name] = normalized
		}
	}
	return requested, details
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
// source images for edits. It returns the first "n" field's value when that
// is a positive integer, 0 otherwise.
func collectMultipartImageRequest(body []byte, contentType string, edit bool, details *converter.ImageBillingDetails) (requested int) {
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil || params["boundary"] == "" {
		return 0
	}
	sawN := false
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	for {
		part, err := reader.NextPart()
		if err != nil {
			return requested
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
			return requested
		}
		if part.FormName() == "n" && !sawN {
			sawN = true
			if n, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil && n > 0 {
				requested = n
			}
		}
		if len(data) > maxImageParamRawLen {
			continue
		}
		if normalized, ok := models.NormalizeImageFormValue(string(data)); ok {
			details.RequestParams[name] = normalized
		}
	}
}

// observeImageResponseBody records what a successful non-streaming image
// response delivered.
func (l *RequestLogContext) observeImageResponseBody(body []byte) {
	facts := imageFactsFromResponseBody(body, l.ImageCount)
	l.imageResponse = &facts
}

// observeImageStreamPayloads records the latest image usage event seen while
// a stream is relayed; streams report it in a usage event that isn't
// necessarily their last data frame.
func (l *RequestLogContext) observeImageStreamPayloads(payloads [][]byte) {
	if facts := imageFactsFromStreamPayloads(payloads); facts.CountKnown || facts.InputImagesKnown {
		l.imageResponse = &facts
	}
}

// finalizeImageUsage sets the image count and per-image pricing inputs of a
// successful image request from the request and whatever its response
// reported; the request's "n" is billed when the response gave no count.
// Failed requests bill no images.
func (l *RequestLogContext) finalizeImageUsage(status string) {
	if !l.IsImageGeneration || status != "success" {
		return
	}
	if l.TokenUsage == nil {
		l.TokenUsage = &converter.TokenUsage{}
	}
	var facts imageResponseFacts
	if l.imageResponse != nil {
		facts = *l.imageResponse
	}

	l.TokenUsage.ImageCount = max(l.ImageCount, 1)
	if facts.CountKnown {
		l.TokenUsage.ImageCount = facts.Count
	}

	if l.imageBillingRequest == nil {
		return
	}
	details := *l.imageBillingRequest
	details.OutputPixels = facts.OutputPixels
	if facts.InputImagesKnown {
		details.InputImages = facts.InputImages
	}
	l.TokenUsage.ImageBilling = &details
}
