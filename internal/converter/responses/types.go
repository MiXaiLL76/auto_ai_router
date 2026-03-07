package responses

// Request types

// Request represents an OpenAI Responses API request.
type Request struct {
	Model           string            `json:"model"`
	Input           interface{}       `json:"input"` // string | []InputItem
	Instructions    string            `json:"instructions,omitempty"`
	MaxOutputTokens *int              `json:"max_output_tokens,omitempty"`
	Temperature     *float64          `json:"temperature,omitempty"`
	TopP            *float64          `json:"top_p,omitempty"`
	Stream          bool              `json:"stream,omitempty"`
	Tools           []Tool            `json:"tools,omitempty"`
	ToolChoice      interface{}       `json:"tool_choice,omitempty"`
	Reasoning       *Reasoning        `json:"reasoning,omitempty"`
	Text            *TextConfig       `json:"text,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
	// Fields we pass through but don't convert
	Store              *bool  `json:"store,omitempty"`
	PreviousResponseID string `json:"previous_response_id,omitempty"`
	ParallelToolCalls  *bool  `json:"parallel_tool_calls,omitempty"`
	ServiceTier        string `json:"service_tier,omitempty"`
	User               string `json:"user,omitempty"`
}

// InputMessage represents a message in the Responses API input array.
type InputMessage struct {
	Type    string      `json:"type,omitempty"` // "message" (may be absent for EasyInputMessage)
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string | []ContentPart
}

// InputFunctionCall represents a function call in the Responses API input.
type InputFunctionCall struct {
	Type      string `json:"type"` // "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// InputFunctionCallOutput represents function call output in the Responses API input.
type InputFunctionCallOutput struct {
	Type   string `json:"type"` // "function_call_output"
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

// ContentPart represents a content part in the Responses API input.
type ContentPart struct {
	Type     string `json:"type"` // "input_text", "input_image", "input_audio"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"` // URL for input_image
	FileID   string `json:"file_id,omitempty"`
	Detail   string `json:"detail,omitempty"`
	Data     string `json:"data,omitempty"`   // base64 for audio
	Format   string `json:"format,omitempty"` // audio format
}

// Tool represents a tool in the Responses API (flat structure).
type Tool struct {
	Type        string      `json:"type"` // "function"
	Name        string      `json:"name,omitempty"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`
}

// Reasoning represents reasoning configuration in the Responses API.
type Reasoning struct {
	Effort  string `json:"effort,omitempty"` // "low","medium","high"
	Summary string `json:"summary,omitempty"`
}

// TextConfig represents text output configuration in the Responses API.
type TextConfig struct {
	Format interface{} `json:"format,omitempty"` // {type: "text"} | {type: "json_schema", ...}
}

// Response types

// Response represents an OpenAI Responses API response.
type Response struct {
	ID                string            `json:"id"`
	Object            string            `json:"object"` // "response"
	CreatedAt         int64             `json:"created_at"`
	Model             string            `json:"model"`
	Status            string            `json:"status"` // "completed","failed","incomplete"
	Output            []OutputItem      `json:"output"`
	Usage             *Usage            `json:"usage,omitempty"`
	Error             interface{}       `json:"error"`              // null or error object
	IncompleteDetails interface{}       `json:"incomplete_details"` // null or {reason: "max_output_tokens"}
	Metadata          map[string]string `json:"metadata"`
	Temperature       *float64          `json:"temperature,omitempty"`
	TopP              *float64          `json:"top_p,omitempty"`
	MaxOutputTokens   *int              `json:"max_output_tokens,omitempty"`
	Tools             []Tool            `json:"tools"`
	ToolChoice        interface{}       `json:"tool_choice,omitempty"`
	ParallelToolCalls bool              `json:"parallel_tool_calls"`
	// Nullable fields
	Instructions       interface{} `json:"instructions"` // null
	Reasoning          interface{} `json:"reasoning,omitempty"`
	Text               interface{} `json:"text,omitempty"`
	Truncation         string      `json:"truncation,omitempty"`
	PreviousResponseID interface{} `json:"previous_response_id"` // null
	Store              bool        `json:"store"`
}

// OutputItem represents an output item in a Responses API response.
type OutputItem struct {
	Type    string          `json:"type"` // "message" | "function_call"
	ID      string          `json:"id"`
	Status  string          `json:"status,omitempty"`  // "completed"
	Role    string          `json:"role,omitempty"`    // "assistant" (for message)
	Content []OutputContent `json:"content,omitempty"` // for message
	// For function_call
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// OutputContent represents content within a message output item.
type OutputContent struct {
	Type        string        `json:"type"` // "output_text"
	Text        string        `json:"text"`
	Annotations []interface{} `json:"annotations"`
}

// Usage represents token usage in a Responses API response.
type Usage struct {
	InputTokens         int            `json:"input_tokens"`
	OutputTokens        int            `json:"output_tokens"`
	TotalTokens         int            `json:"total_tokens"`
	InputTokensDetails  *InputDetails  `json:"input_tokens_details"`
	OutputTokensDetails *OutputDetails `json:"output_tokens_details"`
}

// InputDetails represents details of input token usage.
type InputDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// OutputDetails represents details of output token usage.
type OutputDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}
