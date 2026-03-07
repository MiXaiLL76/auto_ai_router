package converter

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

// DecodeEventStreamToSSE reads an AWS Event Stream binary-framed stream and
// writes standard SSE lines ("data: {json}\n\n") to output.
//
// AWS Event Stream frame layout:
//
//	[4] total byte length          (big-endian uint32)
//	[4] headers byte length        (big-endian uint32)
//	[4] prelude CRC-32             (skipped)
//	[N] headers                    (skipped)
//	[M] payload                    (JSON with base64 "bytes" field)
//	[4] message CRC-32             (skipped)
//
// Each payload is: {"bytes":"<base64-encoded Anthropic event JSON>","p":"<padding>"}
// The decoded bytes are written as SSE data lines for downstream SSE parsers.
func DecodeEventStreamToSSE(reader io.Reader, writer io.Writer) error {
	var prelude [12]byte

	for {
		// Read 12-byte prelude: total_length(4) + headers_length(4) + prelude_crc(4)
		if _, err := io.ReadFull(reader, prelude[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil // Stream ended
			}
			return fmt.Errorf("event stream: read prelude: %w", err)
		}

		totalLength := binary.BigEndian.Uint32(prelude[0:4])
		headersLength := binary.BigEndian.Uint32(prelude[4:8])

		// Remaining bytes after prelude: headers + payload + message_crc(4)
		remaining := int(totalLength) - 12
		if remaining <= 0 {
			continue
		}

		frame := make([]byte, remaining)
		if _, err := io.ReadFull(reader, frame); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil
			}
			return fmt.Errorf("event stream: read frame: %w", err)
		}

		// Payload starts after headers, ends before message CRC (last 4 bytes)
		payloadStart := int(headersLength)
		payloadEnd := len(frame) - 4
		if payloadStart >= payloadEnd {
			continue
		}
		payload := frame[payloadStart:payloadEnd]

		// Parse {"bytes":"<base64>", "p":"..."}
		var envelope struct {
			Bytes string `json:"bytes"`
		}
		if err := json.Unmarshal(payload, &envelope); err != nil || envelope.Bytes == "" {
			continue
		}

		decoded, err := base64.StdEncoding.DecodeString(envelope.Bytes)
		if err != nil {
			continue
		}

		// Emit as SSE line
		if _, err := fmt.Fprintf(writer, "data: %s\n\n", decoded); err != nil {
			return fmt.Errorf("event stream: write SSE: %w", err)
		}
	}
}
