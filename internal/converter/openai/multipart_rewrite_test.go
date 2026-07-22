package openai

import (
	"bytes"
	"io"
	"mime"
	"mime/multipart"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makePNG returns minimal valid PNG magic bytes (89 50 4E 47 ...).
func makePNG() []byte {
	return []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00}
}

// makeJPEG returns minimal JPEG magic bytes.
func makeJPEG() []byte {
	return []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00}
}

// makeWEBP returns minimal WebP magic bytes.
func makeWEBP() []byte {
	b := make([]byte, 12)
	copy(b[0:], []byte("RIFF"))
	copy(b[4:], []byte{0x00, 0x00, 0x00, 0x00}) // file size (ignored)
	copy(b[8:], []byte("WEBP"))
	return b
}

// buildMultipart creates a multipart/form-data body with the given fields.
// fields is a slice of (name, contentType, data) triples.
func buildMultipart(t *testing.T, fields []struct {
	name        string
	contentType string
	filename    string
	data        []byte
}) (body []byte, contentType string) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, f := range fields {
		var pw io.Writer
		var err error
		if f.contentType != "" || f.filename != "" {
			h := make(map[string][]string)
			cd := `form-data; name="` + f.name + `"`
			if f.filename != "" {
				cd += `; filename="` + f.filename + `"`
			}
			h["Content-Disposition"] = []string{cd}
			if f.contentType != "" {
				h["Content-Type"] = []string{f.contentType}
			}
			pw, err = w.CreatePart(h)
		} else {
			pw, err = w.CreateFormField(f.name)
		}
		if err != nil {
			t.Fatalf("CreatePart(%q): %v", f.name, err)
		}
		if _, err := pw.Write(f.data); err != nil {
			t.Fatalf("Write(%q): %v", f.name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("multipart close: %v", err)
	}
	return buf.Bytes(), w.FormDataContentType()
}

// parseResult parses the rewritten multipart body and returns a map of
// field name → (contentType, data).
func parseResult(t *testing.T, body []byte, contentType string) map[string]struct {
	contentType string
	filename    string
	data        []byte
} {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("ParseMediaType: %v", err)
	}
	r := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	result := make(map[string]struct {
		contentType string
		filename    string
		data        []byte
	})
	for {
		p, err := r.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("NextPart: %v", err)
		}
		data, _ := io.ReadAll(p)
		_ = p.Close()
		result[p.FormName()] = struct {
			contentType string
			filename    string
			data        []byte
		}{
			contentType: p.Header.Get("Content-Type"),
			filename:    p.FileName(),
			data:        data,
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// detectImageMIME
// ---------------------------------------------------------------------------

func TestDetectImageMIME(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		wantMIME string
		wantOK   bool
	}{
		{"png", makePNG(), "image/png", true},
		{"jpeg", makeJPEG(), "image/jpeg", true},
		{"webp", makeWEBP(), "image/webp", true},
		{"octet-stream random", []byte{0x00, 0x01, 0x02, 0x03}, "", false},
		{"empty", []byte{}, "", false},
		{"too short for webp", []byte("RIFF"), "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mime, ok := detectImageMIME(tc.data)
			if ok != tc.wantOK {
				t.Errorf("ok = %v, want %v", ok, tc.wantOK)
			}
			if mime != tc.wantMIME {
				t.Errorf("mime = %q, want %q", mime, tc.wantMIME)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// isImageFieldName
// ---------------------------------------------------------------------------

func TestIsImageFieldName(t *testing.T) {
	yes := []string{"image", "mask", "image[0]", "image[1]", "image[99]"}
	no := []string{"model", "prompt", "n", "size", "response_format", "imageX", "image[]extra"}

	for _, name := range yes {
		if !isImageFieldName(name) {
			t.Errorf("isImageFieldName(%q) = false, want true", name)
		}
	}
	for _, name := range no {
		if isImageFieldName(name) {
			t.Errorf("isImageFieldName(%q) = true, want false", name)
		}
	}
}

// ---------------------------------------------------------------------------
// mimeTypeToExt
// ---------------------------------------------------------------------------

func TestMimeTypeToExt(t *testing.T) {
	cases := map[string]string{
		"image/png":  ".png",
		"image/jpeg": ".jpg",
		"image/webp": ".webp",
		"image/gif":  "",
		"":           "",
	}
	for mime, want := range cases {
		if got := mimeTypeToExt(mime); got != want {
			t.Errorf("mimeTypeToExt(%q) = %q, want %q", mime, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// RewriteImageEditMultipart — invalid / passthrough cases
// ---------------------------------------------------------------------------

func TestRewriteImageEditMultipart_ReplacesModel(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"model", "", "", []byte("gpt-image-2-vsellm")},
		{"prompt", "", "", []byte("make it blue")},
		{"image", "image/png", "input.png", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "gpt-image-2", false)
	parts := parseResult(t, newBody, newCT)

	if got := string(parts["model"].data); got != "gpt-image-2" {
		t.Errorf("model = %q, want %q", got, "gpt-image-2")
	}
	if got := string(parts["prompt"].data); got != "make it blue" {
		t.Errorf("prompt = %q, want %q", got, "make it blue")
	}
	if !bytes.Equal(parts["image"].data, makePNG()) {
		t.Error("image data changed")
	}
}

func TestRewriteImageEditMultipart_InvalidContentType(t *testing.T) {
	body := []byte("not-multipart-data")
	got, gotCT := RewriteImageEditMultipart(body, "application/json", "", false)
	if !bytes.Equal(got, body) {
		t.Error("expected original body returned for non-multipart content-type")
	}
	if gotCT != "application/json" {
		t.Errorf("expected original content-type, got %q", gotCT)
	}
}

func TestRewriteImageEditMultipart_NoBoundary(t *testing.T) {
	body := []byte("data")
	ct := "multipart/form-data"
	got, gotCT := RewriteImageEditMultipart(body, ct, "", false)
	if !bytes.Equal(got, body) {
		t.Error("expected original body returned when no boundary")
	}
	if gotCT != ct {
		t.Errorf("expected original content-type, got %q", gotCT)
	}
}

// ---------------------------------------------------------------------------
// RewriteImageEditMultipart — MIME type detection
// ---------------------------------------------------------------------------

func TestRewriteImageEditMultipart_FixesPNGMIME(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"model", "", "", []byte("gpt-image-1-mini")},
		{"prompt", "", "", []byte("edit this")},
		{"image", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)

	parts := parseResult(t, newBody, newCT)
	img, ok := parts["image"]
	if !ok {
		t.Fatal("image field missing from rewritten body")
	}
	if img.contentType != "image/png" {
		t.Errorf("Content-Type = %q, want %q", img.contentType, "image/png")
	}
	if img.filename != "image.png" {
		t.Errorf("filename = %q, want %q", img.filename, "image.png")
	}
	if !bytes.Equal(img.data, makePNG()) {
		t.Error("image data was corrupted")
	}
}

func TestRewriteImageEditMultipart_FixesJPEGMIME(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", makeJPEG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if parts["image"].contentType != "image/jpeg" {
		t.Errorf("Content-Type = %q, want image/jpeg", parts["image"].contentType)
	}
	if parts["image"].filename != "image.jpg" {
		t.Errorf("filename = %q, want image.jpg", parts["image"].filename)
	}
}

func TestRewriteImageEditMultipart_FixesWEBPMIME(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", makeWEBP()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if parts["image"].contentType != "image/webp" {
		t.Errorf("Content-Type = %q, want image/webp", parts["image"].contentType)
	}
}

func TestRewriteImageEditMultipart_PreservesExplicitPNGMIME(t *testing.T) {
	// If the client already sends image/png, it must not be changed.
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "image/png", "photo.png", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if parts["image"].contentType != "image/png" {
		t.Errorf("Content-Type = %q, want image/png", parts["image"].contentType)
	}
	if parts["image"].filename != "photo.png" {
		t.Errorf("filename = %q, want photo.png", parts["image"].filename)
	}
}

func TestRewriteImageEditMultipart_UnknownImageData_NotChanged(t *testing.T) {
	// Unknown bytes with octet-stream — MIME can't be detected, part kept as-is.
	unknownData := []byte{0x00, 0x01, 0x02, 0x03}
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", unknownData},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	// Content-Type stays application/octet-stream (no detection possible).
	if parts["image"].contentType != "application/octet-stream" {
		t.Errorf("Content-Type = %q, want application/octet-stream", parts["image"].contentType)
	}
	if !bytes.Equal(parts["image"].data, unknownData) {
		t.Error("image data was corrupted")
	}
}

func TestRewriteImageEditMultipart_MultipleImages_RenamedToArraySyntax(t *testing.T) {
	// Python SDK sends multiple "image" fields; OpenAI requires "image[]".
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", makePNG()},
		{"image", "application/octet-stream", "", makeJPEG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)

	// Count occurrences of "image" vs "image[]" in the rewritten body.
	_, newParams, _ := mime.ParseMediaType(newCT)
	r := multipart.NewReader(bytes.NewReader(newBody), newParams["boundary"])
	imageCount, imageArrayCount := 0, 0
	for {
		p, err := r.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("NextPart: %v", err)
		}
		_, _ = io.ReadAll(p)
		_ = p.Close()
		switch p.FormName() {
		case "image":
			imageCount++
		case "image[]":
			imageArrayCount++
		}
	}

	if imageCount != 0 {
		t.Errorf("found %d 'image' fields, want 0 (should all be renamed to 'image[]')", imageCount)
	}
	if imageArrayCount != 2 {
		t.Errorf("found %d 'image[]' fields, want 2", imageArrayCount)
	}
}

func TestRewriteImageEditMultipart_SingleImage_KeepsOriginalName(t *testing.T) {
	// Single image must NOT be renamed to image[].
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if _, exists := parts["image"]; !exists {
		t.Error("single 'image' field should keep its name, not be renamed to 'image[]'")
	}
	if _, exists := parts["image[]"]; exists {
		t.Error("single image should not produce 'image[]' field")
	}
}

func TestRewriteImageEditMultipart_MultipleImages_MIMEFixed(t *testing.T) {
	// image[0] and image[1] already using array syntax — MIME types still get corrected.
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image[0]", "application/octet-stream", "", makePNG()},
		{"image[1]", "application/octet-stream", "", makeJPEG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if parts["image[0]"].contentType != "image/png" {
		t.Errorf("image[0] Content-Type = %q, want image/png", parts["image[0]"].contentType)
	}
	if parts["image[1]"].contentType != "image/jpeg" {
		t.Errorf("image[1] Content-Type = %q, want image/jpeg", parts["image[1]"].contentType)
	}
}

func TestRewriteImageEditMultipart_MaskField(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"mask", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if parts["mask"].contentType != "image/png" {
		t.Errorf("mask Content-Type = %q, want image/png", parts["mask"].contentType)
	}
}

// ---------------------------------------------------------------------------
// RewriteImageEditMultipart — response_format stripping
// ---------------------------------------------------------------------------

func TestRewriteImageEditMultipart_StripResponseFormat(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"model", "", "", []byte("gpt-image-1-mini")},
		{"prompt", "", "", []byte("edit this")},
		{"response_format", "", "", []byte("b64_json")},
		{"image", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", true)
	parts := parseResult(t, newBody, newCT)

	if _, exists := parts["response_format"]; exists {
		t.Error("response_format should have been stripped but was found in rewritten body")
	}
	if _, exists := parts["model"]; !exists {
		t.Error("model field should be preserved")
	}
	if _, exists := parts["prompt"]; !exists {
		t.Error("prompt field should be preserved")
	}
}

func TestRewriteImageEditMultipart_PreserveResponseFormat_WhenNotStripping(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"response_format", "", "", []byte("b64_json")},
		{"image", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	if _, exists := parts["response_format"]; !exists {
		t.Error("response_format should be preserved when stripResponseFormat=false")
	}
}

// ---------------------------------------------------------------------------
// RewriteImageEditMultipart — data integrity
// ---------------------------------------------------------------------------

func TestRewriteImageEditMultipart_AllFieldsPreserved(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"model", "", "", []byte("gpt-image-1-mini")},
		{"prompt", "", "", []byte("make it blue")},
		{"n", "", "", []byte("1")},
		{"size", "", "", []byte("1024x1024")},
		{"image", "application/octet-stream", "", makePNG()},
	})

	newBody, newCT := RewriteImageEditMultipart(body, ct, "", false)
	parts := parseResult(t, newBody, newCT)

	for _, field := range []string{"model", "prompt", "n", "size", "image"} {
		if _, ok := parts[field]; !ok {
			t.Errorf("field %q missing from rewritten body", field)
		}
	}
	if string(parts["model"].data) != "gpt-image-1-mini" {
		t.Errorf("model value corrupted: %q", parts["model"].data)
	}
	if string(parts["prompt"].data) != "make it blue" {
		t.Errorf("prompt value corrupted: %q", parts["prompt"].data)
	}
}

func TestRewriteImageEditMultipart_NewBoundaryDiffersFromOriginal(t *testing.T) {
	body, ct := buildMultipart(t, []struct {
		name        string
		contentType string
		filename    string
		data        []byte
	}{
		{"image", "application/octet-stream", "", makePNG()},
	})

	_, newCT := RewriteImageEditMultipart(body, ct, "", false)

	_, origParams, _ := mime.ParseMediaType(ct)
	_, newParams, _ := mime.ParseMediaType(newCT)

	if origParams["boundary"] == newParams["boundary"] {
		t.Error("new Content-Type should use a different boundary than the original")
	}
	if !strings.HasPrefix(newCT, "multipart/form-data") {
		t.Errorf("new Content-Type should be multipart/form-data, got %q", newCT)
	}
}
