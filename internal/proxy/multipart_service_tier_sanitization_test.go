package proxy

import (
	"bytes"
	"io"
	"mime"
	"mime/multipart"
	"net/textproto"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type multipartTestPart struct {
	disposition string
	contentType string
	headers     textproto.MIMEHeader
	data        []byte
}

type parsedMultipartTestPart struct {
	name     string
	filename string
	header   textproto.MIMEHeader
	data     []byte
}

func buildMultipartTestBody(t *testing.T, boundary string, parts ...multipartTestPart) ([]byte, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.SetBoundary(boundary))
	for _, spec := range parts {
		header := cloneMIMEHeader(spec.headers)
		if header == nil {
			header = make(textproto.MIMEHeader)
		}
		header.Set("Content-Disposition", spec.disposition)
		if spec.contentType != "" {
			header.Set("Content-Type", spec.contentType)
		}
		part, err := writer.CreatePart(header)
		require.NoError(t, err)
		_, err = part.Write(spec.data)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Close())
	return body.Bytes(), writer.FormDataContentType()
}

func parseMultipartTestBody(t *testing.T, body []byte, contentType string) (string, []parsedMultipartTestPart) {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(contentType)
	require.NoError(t, err)
	require.Equal(t, "multipart/form-data", mediaType)
	boundary := params["boundary"]
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var parts []parsedMultipartTestPart
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		data, err := io.ReadAll(part)
		require.NoError(t, err)
		parts = append(parts, parsedMultipartTestPart{
			name:     part.FormName(),
			filename: part.FileName(),
			header:   cloneMIMEHeader(part.Header),
			data:     data,
		})
		require.NoError(t, part.Close())
	}
	return boundary, parts
}

func TestSanitizeMultipartRequestBodyRemovesEveryClientServiceTierForm(t *testing.T) {
	const boundary = "air-security-boundary-123"
	imageOne := []byte{0x89, 'P', 'N', 'G', 0, 1, 2, 3}
	imageTwo := []byte{0xff, 0xd8, 0xff, 0x00, 0x02}
	mask := []byte{0x89, 'P', 'N', 'G', 9, 8, 7}
	body, contentType := buildMultipartTestBody(t, boundary,
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: []byte("priority")},
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-image-1")},
		multipartTestPart{
			disposition: `form-data; name="image"; filename*=UTF-8''input-%E2%82%AC.png`,
			contentType: "image/png",
			headers: textproto.MIMEHeader{
				"Content-Id": {"image-one"},
				"X-Custom":   {"first", "second"},
			},
			data: imageOne,
		},
		multipartTestPart{disposition: `form-data; name="extra_body"`, data: []byte(`{"service_tier":"default","other":"value","large":9223372036854775807}`)},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte(`keep {"service_tier":"priority"} here`)},
		multipartTestPart{disposition: `form-data; name="extra_body[service_tier]"`, data: []byte("flex")},
		multipartTestPart{disposition: `form-data; name="image"; filename="second.jpg"`, contentType: "image/jpeg", data: imageTwo},
		multipartTestPart{disposition: `form-data; name="extra_body.service_tier"`, data: []byte("priority")},
		multipartTestPart{disposition: `form-data; name="mask"; filename="mask.png"`, contentType: "image/png", data: mask},
		multipartTestPart{disposition: `form-data; name="service_tier"; filename="tier.bin"`, contentType: "application/octet-stream", data: []byte{0, 1, 2, 3}},
		multipartTestPart{disposition: `form-data; name*=UTF-8''service_tier`, data: []byte("default")},
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: nil},
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: []byte(`{"tier":"priority"}`)},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte("duplicate prompt")},
	)

	result, err := sanitizeAndExtractRequestBody(body, contentType)
	require.NoError(t, err)
	require.True(t, result.Changed)
	assert.Equal(t, "gpt-image-1", result.ModelID)

	gotBoundary, parts := parseMultipartTestBody(t, result.Body, contentType)
	assert.Equal(t, boundary, gotBoundary)
	require.Equal(t, []string{"model", "image", "extra_body", "prompt", "image", "mask", "prompt"}, multipartPartNames(parts))

	assert.Equal(t, imageOne, parts[1].data)
	assert.Equal(t, "input-€.png", parts[1].filename)
	assert.Equal(t, "image/png", parts[1].header.Get("Content-Type"))
	assert.Equal(t, "image-one", parts[1].header.Get("Content-Id"))
	assert.Equal(t, []string{"first", "second"}, parts[1].header.Values("X-Custom"))
	assert.JSONEq(t, `{"other":"value","large":9223372036854775807}`, string(parts[2].data))
	assert.Contains(t, string(parts[3].data), `"service_tier"`)
	assert.Equal(t, imageTwo, parts[4].data)
	assert.Equal(t, "second.jpg", parts[4].filename)
	assert.Equal(t, mask, parts[5].data)
	assert.Equal(t, "mask.png", parts[5].filename)
}

func multipartPartNames(parts []parsedMultipartTestPart) []string {
	names := make([]string, len(parts))
	for i := range parts {
		names[i] = parts[i].name
	}
	return names
}

func TestSanitizeMultipartRequestBodyReturnsOriginalBytesWhenUnchanged(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, "unchanged-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="prompt"`, data: []byte(`service_tier=priority`)},
		multipartTestPart{disposition: `form-data; name="image"; filename="input.bin"`, contentType: "application/octet-stream", data: []byte(`{"service_tier":"priority"}`)},
	)

	result, err := sanitizeAndExtractRequestBody(body, contentType)
	require.NoError(t, err)
	assert.False(t, result.Changed)
	assert.Equal(t, body, result.Body)
	assert.Equal(t, "gpt-4", result.ModelID)
}

func TestSanitizeMultipartRequestBodyRejectsMalformedInput(t *testing.T) {
	validBody, validContentType := buildMultipartTestBody(t, "malformed-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="image"; filename="input.png"`, contentType: "image/png", data: []byte{0x89, 'P', 'N', 'G'}},
		multipartTestPart{disposition: `form-data; name="service_tier"`, data: []byte("priority")},
	)
	invalidExtraBody, invalidExtraContentType := buildMultipartTestBody(t, "invalid-extra-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="extra_body"`, data: []byte(`{"service_tier":`)},
	)
	invalidDisposition, invalidDispositionContentType := buildMultipartTestBody(t, "invalid-disposition-boundary",
		multipartTestPart{disposition: `form-data; name="model`, data: []byte("gpt-4")},
	)

	tests := []struct {
		name        string
		body        []byte
		contentType string
	}{
		{name: "missing boundary", body: validBody, contentType: "multipart/form-data"},
		{name: "invalid boundary parameter", body: validBody, contentType: `multipart/form-data; boundary="unterminated`},
		{name: "unsupported boundary characters", body: validBody, contentType: `multipart/form-data; boundary="bad☃boundary"`},
		{name: "truncated after model", body: validBody[:bytes.Index(validBody, []byte("gpt-4"))+len("gpt-4")], contentType: validContentType},
		{name: "truncated after file", body: validBody[:bytes.Index(validBody, []byte{0x89, 'P', 'N', 'G'})+4], contentType: validContentType},
		{name: "truncated after model and file", body: validBody[:len(validBody)-12], contentType: validContentType},
		{name: "truncated before service tier", body: validBody[:bytes.Index(validBody, []byte("service_tier"))-5], contentType: validContentType},
		{name: "invalid extra body object", body: invalidExtraBody, contentType: invalidExtraContentType},
		{name: "invalid content disposition", body: invalidDisposition, contentType: invalidDispositionContentType},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := sanitizeAndExtractRequestBody(tt.body, tt.contentType)
			assert.ErrorIs(t, err, errInvalidMultipartRequestBody)
			assert.Empty(t, result.Body)
		})
	}
}

func TestSanitizeRequestBodyUsesParsedMediaType(t *testing.T) {
	body := []byte(`{"model":"gpt-4","service_tier":"priority"}`)
	result, err := sanitizeAndExtractRequestBody(body, "application/json; profile=multipart/form-data")
	require.NoError(t, err)
	assert.True(t, result.Changed)
	assert.NotContains(t, string(result.Body), `"service_tier"`)
}

func TestMultipartPartNameMatchingIsNotRecursiveOrCaseInsensitive(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, "name-matching-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-4")},
		multipartTestPart{disposition: `form-data; name="metadata.service_tier"`, data: []byte("keep")},
		multipartTestPart{disposition: `form-data; name="Service_Tier"`, data: []byte("keep-case")},
	)
	result, err := sanitizeAndExtractRequestBody(body, contentType)
	require.NoError(t, err)
	assert.False(t, result.Changed)
	_, parts := parseMultipartTestBody(t, result.Body, contentType)
	assert.Equal(t, []string{"model", "metadata.service_tier", "Service_Tier"}, multipartPartNames(parts))
	assert.True(t, strings.Contains(string(parts[1].data), "keep"))
}
