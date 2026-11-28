package vertex

import (
	"encoding/base64"
	"fmt"
	"mime"
	"strings"

	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"google.golang.org/genai"
)

// mimeOverride returns an explicit MIME type from mime_type or format on the
// media object or the part itself. format may be a MIME type or an extension.
func mimeOverride(objects ...map[string]interface{}) string {
	for _, obj := range objects {
		if obj == nil {
			continue
		}
		for _, key := range []string{"mime_type", "mimeType", "format"} {
			value, _ := obj[key].(string)
			value = strings.TrimSpace(value)
			if value == "" {
				continue
			}
			if strings.Contains(value, "/") {
				return value
			}
			if mimeType := mimeTypeFromName("file." + value); mimeType != "" {
				return mimeType
			}
		}
	}
	return ""
}

// audioExtensions extends mimeTypeMap with the audio containers Gemini
// accepts in file parts; kept out of mimeTypeMap so chat URL handling is
// unchanged.
var audioExtensions = map[string]string{
	"mp3": "audio/mpeg",
	"wav": "audio/wav",
}

func mimeTypeFromName(name string) string {
	if mimeType := getMimeTypeFromURL(name); mimeType != "" {
		return mimeType
	}
	ext := ""
	if idx := strings.LastIndex(name, "."); idx >= 0 {
		ext = strings.ToLower(name[idx+1:])
	}
	return audioExtensions[ext]
}

func mediaPart(ref, override, defaultMIME, param string) (*genai.Part, error) {
	if ref == "" {
		return nil, converterutil.NewRequestValidationError(param, "media part requires a url")
	}
	if strings.HasPrefix(ref, "data:") {
		part, err := parseDataURLToPart(ref)
		if err != nil {
			return nil, err
		}
		if part == nil {
			return nil, converterutil.NewRequestValidationError(param, "invalid or empty base64 data URL")
		}
		if override != "" {
			part.InlineData.MIMEType = override
		}
		return part, nil
	}
	mimeType := override
	if mimeType == "" {
		mimeType = mimeTypeFromName(stripURLQuery(ref))
	}
	if mimeType == "" {
		mimeType = defaultMIME
	}
	if mimeType == "" {
		return nil, converterutil.NewRequestValidationError(param, "cannot determine the MIME type of the file; set mime_type")
	}
	part, err := parseURLToPart(ref, map[string]interface{}{"format": mimeType}, param)
	if err != nil {
		return nil, err
	}
	if part == nil {
		return nil, converterutil.NewRequestValidationError(param, "unsupported URL: use a data: URL, a public https:// URL or a gs:// URI")
	}
	return part, nil
}

func stripURLQuery(ref string) string {
	if idx := strings.IndexAny(ref, "?#"); idx >= 0 {
		return ref[:idx]
	}
	return ref
}

func inlineBase64Part(data, mimeType, param string) (*genai.Part, error) {
	if len(data) > maxBase64Size {
		return nil, converterutil.NewRequestEntityTooLargeError(param, fmt.Sprintf("inline payload exceeds %dMB limit", maxBase64Size/(1024*1024)))
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, converterutil.NewRequestValidationError(param, "data is not valid base64")
	}
	if len(decoded) == 0 {
		return nil, converterutil.NewRequestValidationError(param, "data is empty")
	}
	return &genai.Part{InlineData: &genai.Blob{MIMEType: mimeType, Data: decoded}}, nil
}

// FileDataPart converts an OpenAI file_data value into an inline part: a data:
// URL, or bare base64 whose MIME type then has to come from mimeType or the
// file name. An explicit mimeType wins over the one in the data: URL.
func FileDataPart(fileData, mimeType, filename, param string) (*genai.Part, error) {
	var part *genai.Part
	var err error
	if strings.HasPrefix(fileData, "data:") {
		part, err = mediaPart(fileData, mimeType, "", param)
	} else {
		if mimeType == "" && filename != "" {
			mimeType = mimeTypeFromName(filename)
		}
		if mimeType == "" {
			return nil, converterutil.NewRequestValidationError(param, "bare base64 file_data requires mime_type (or a data: URL)")
		}
		part, err = inlineBase64Part(fileData, mimeType, param)
	}
	if err != nil {
		return nil, err
	}
	// "data:;base64,..." parses with ";base64" as its MIME type.
	if !isMediaType(part.InlineData.MIMEType) {
		return nil, converterutil.NewRequestValidationError(param, "file_data has no valid MIME type; set mime_type")
	}
	return part, nil
}

// isMediaType reports whether value is a type/subtype MIME type.
func isMediaType(value string) bool {
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return false
	}
	typ, subtype, ok := strings.Cut(mediaType, "/")
	return ok && typ != "" && subtype != ""
}
