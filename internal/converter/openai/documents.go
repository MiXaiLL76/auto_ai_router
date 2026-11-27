package openai

import (
	"bytes"
	"encoding/json"
)

// The router's Chat format carries a PDF as the Anthropic-shaped part
//
//	{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"..."}}
//
// (what /v1/messages hands over, see docs/getting-started/api.md). OpenAI Chat
// Completions has no such part: the same bytes go as
//
//	{"type":"file","file":{"file_data":"data:application/pdf;base64,...","filename":"..."}}
//
// A document given by URL has no Chat Completions form at all and is left as it
// is, for the provider to refuse.

// DefaultDocumentFilename names a document whose source format has no file
// name: OpenAI rejects file_data without one.
const DefaultDocumentFilename = "document.pdf"

var documentNeedle = []byte(`"document"`)

// DocumentFile returns the OpenAI file object for a base64 document part: the
// data: URL and the document title as the file name. ok is false for any
// other source.
func DocumentFile(part map[string]interface{}) (file map[string]interface{}, ok bool) {
	source, _ := part["source"].(map[string]interface{})
	if source == nil || source["type"] != "base64" {
		return nil, false
	}
	mediaType, _ := source["media_type"].(string)
	data, _ := source["data"].(string)
	if mediaType == "" || data == "" {
		return nil, false
	}
	filename, _ := part["title"].(string)
	if filename == "" {
		filename = DefaultDocumentFilename
	}
	return map[string]interface{}{
		"file_data": "data:" + mediaType + ";base64," + data,
		"filename":  filename,
	}, true
}

// ConvertDocumentParts rewrites the base64 document parts of a Chat
// Completions body into file parts. cache_control is not carried over: OpenAI
// rejects it on a file part. Tool messages are left alone. Bodies without such
// a part are returned unchanged.
func ConvertDocumentParts(body []byte) []byte {
	if !bytes.Contains(body, documentNeedle) {
		return body
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(data["messages"], &messages); err != nil {
		return body
	}

	changed := false
	for i, rawMessage := range messages {
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil {
			continue
		}
		var role string
		if err := json.Unmarshal(message["role"], &role); err != nil || role == "tool" {
			continue
		}
		var content []json.RawMessage
		if err := json.Unmarshal(message["content"], &content); err != nil {
			continue
		}
		contentChanged := false
		for j, rawPart := range content {
			// Skip the other parts undecoded: they may carry megabytes of base64.
			if !bytes.Contains(rawPart, documentNeedle) {
				continue
			}
			var part map[string]interface{}
			if err := json.Unmarshal(rawPart, &part); err != nil || part["type"] != "document" {
				continue
			}
			file, ok := DocumentFile(part)
			if !ok {
				continue
			}
			converted, err := marshalWithoutHTMLEscape(map[string]interface{}{"type": "file", "file": file})
			if err != nil {
				continue
			}
			content[j] = converted
			contentChanged = true
		}
		if !contentChanged {
			continue
		}
		encodedContent, err := marshalWithoutHTMLEscape(content)
		if err != nil {
			continue
		}
		message["content"] = encodedContent
		encodedMessage, err := marshalWithoutHTMLEscape(message)
		if err != nil {
			continue
		}
		messages[i] = encodedMessage
		changed = true
	}
	if !changed {
		return body
	}
	encodedMessages, err := marshalWithoutHTMLEscape(messages)
	if err != nil {
		return body
	}
	data["messages"] = encodedMessages
	result, err := marshalWithoutHTMLEscape(data)
	if err != nil {
		return body
	}
	return result
}
