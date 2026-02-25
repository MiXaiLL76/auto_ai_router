package anthropic

// AnthropicRequest represents a request to the Anthropic Messages API.
// Serialized directly to JSON for HTTP requests (no SDK dependency).
type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	System        interface{}        `json:"system,omitempty"` // string or []ContentBlock
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Tools         []AnthropicTool    `json:"tools,omitempty"`
	ToolChoice    interface{}        `json:"tool_choice,omitempty"`
	Thinking      *AnthropicThinking `json:"thinking,omitempty"`
	Metadata      *AnthropicMetadata `json:"metadata,omitempty"`
}

// AnthropicMessage represents a single message in the Anthropic conversation.
type AnthropicMessage struct {
	Role    string      `json:"role"`    // "user" or "assistant"
	Content interface{} `json:"content"` // string or []ContentBlock
}

// ContentBlock is a universal content block used in both requests and responses.
type ContentBlock struct {
	Type string `json:"type"`

	// text block
	Text string `json:"text,omitempty"`

	// image / document block
	Source *MediaSource `json:"source,omitempty"`

	// tool_use block (in assistant messages / responses)
	ID    string      `json:"id,omitempty"`
	Name  string      `json:"name,omitempty"`
	Input interface{} `json:"input,omitempty"`

	// tool_result block (in user messages)
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   interface{} `json:"content,omitempty"` // string or []ContentBlock

	// thinking block (in responses)
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// MediaSource describes the source of an image or document content block.
type MediaSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/jpeg", "application/pdf"
	Data      string `json:"data,omitempty"`       // base64-encoded data (type=base64)
	URL       string `json:"url,omitempty"`        // remote URL (type=url)
}

// AnthropicTool represents a tool definition in an Anthropic request.
// Covers both function tools and Anthropic built-in tools (computer_use, bash, etc.).
type AnthropicTool struct {
	// Standard function tool fields
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema,omitempty"`

	// Special Anthropic built-in tool type identifier (e.g. "computer_20241022")
	Type string `json:"type,omitempty"`

	// computer_use specific dimensions
	DisplayWidthPx  int `json:"display_width_px,omitempty"`
	DisplayHeightPx int `json:"display_height_px,omitempty"`
}

// AnthropicThinking controls extended thinking / reasoning in Anthropic models.
type AnthropicThinking struct {
	Type         string `json:"type"`                    // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"` // token budget when enabled
}

// AnthropicMetadata carries per-request metadata sent to Anthropic.
type AnthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

// AnthropicResponse represents a non-streaming response from the Anthropic Messages API.
type AnthropicResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage `json:"usage"`
}

// CacheCreationDetails contains per-TTL breakdown of cache creation tokens.
type CacheCreationDetails struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens,omitempty"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens,omitempty"`
}

// ServerToolUsageDetails holds counts of server-side tool invocations.
type ServerToolUsageDetails struct {
	WebSearchRequests int `json:"web_search_requests,omitempty"`
}

// AnthropicUsage represents token usage reported by the Anthropic API.
// See: https://platform.claude.com/docs/en/api/messages
type AnthropicUsage struct {
	InputTokens              int                     `json:"input_tokens"`
	OutputTokens             int                     `json:"output_tokens"`
	CacheCreationInputTokens int                     `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                     `json:"cache_read_input_tokens,omitempty"`
	CacheCreation            *CacheCreationDetails   `json:"cache_creation,omitempty"`
	ServerToolUse            *ServerToolUsageDetails `json:"server_tool_use,omitempty"`
	ServiceTier              string                  `json:"service_tier,omitempty"`
	InferenceGeo             string                  `json:"inference_geo,omitempty"`
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

// AnthropicStreamEvent represents a single SSE event from an Anthropic streaming response.
type AnthropicStreamEvent struct {
	Type string `json:"type"`

	// message_start: carries the initial message skeleton with input usage
	Message *AnthropicStreamMessage `json:"message,omitempty"`

	// content_block_start: describes the opening block
	Index        int           `json:"index,omitempty"`
	ContentBlock *ContentBlock `json:"content_block,omitempty"`

	// content_block_delta / message_delta: carries incremental data
	Delta *AnthropicStreamDelta `json:"delta,omitempty"`

	// message_delta usage (output_tokens)
	Usage *AnthropicStreamUsage `json:"usage,omitempty"`
}

// AnthropicStreamMessage is the message skeleton delivered in the message_start event.
type AnthropicStreamMessage struct {
	ID    string         `json:"id"`
	Usage AnthropicUsage `json:"usage"`
}

// AnthropicStreamDelta represents incremental data in content_block_delta or message_delta events.
type AnthropicStreamDelta struct {
	Type string `json:"type"`

	// text_delta
	Text string `json:"text,omitempty"`

	// thinking_delta
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// input_json_delta (tool input streaming)
	PartialJSON string `json:"partial_json,omitempty"`

	// message_delta
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// AnthropicStreamUsage carries token counts in message_delta events.
type AnthropicStreamUsage struct {
	InputTokens              int `json:"input_tokens,omitempty"`
	OutputTokens             int `json:"output_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}
