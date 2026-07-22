package responses

import "encoding/json"

// ===== Request Types =====

// Request represents an OpenAI Responses API request.
type Request struct {
	Model                string            `json:"model"`
	Input                interface{}       `json:"input"`                  // string | []InputItem
	Instructions         interface{}       `json:"instructions,omitempty"` // string | nil
	MaxOutputTokens      *int              `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int              `json:"max_tool_calls,omitempty"`
	Temperature          *float64          `json:"temperature,omitempty"`
	TopP                 *float64          `json:"top_p,omitempty"`
	PresencePenalty      *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty     *float64          `json:"frequency_penalty,omitempty"`
	TopLogprobs          *int              `json:"top_logprobs,omitempty"`
	Stop                 interface{}       `json:"stop,omitempty"` // string | []string
	Stream               bool              `json:"stream,omitempty"`
	StreamOptions        interface{}       `json:"stream_options,omitempty"`
	Background           bool              `json:"background,omitempty"`
	Tools                []Tool            `json:"tools,omitempty"`
	ToolChoice           interface{}       `json:"tool_choice,omitempty"`
	Reasoning            *Reasoning        `json:"reasoning,omitempty"`
	Text                 *TextConfig       `json:"text,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
	Conversation         interface{}       `json:"conversation,omitempty"`
	Include              []string          `json:"include,omitempty"`
	Truncation           string            `json:"truncation,omitempty"`
	SafetyIdentifier     string            `json:"safety_identifier,omitempty"`
	Store                *bool             `json:"store,omitempty"`
	PreviousResponseID   string            `json:"previous_response_id,omitempty"`
	ParallelToolCalls    *bool             `json:"parallel_tool_calls,omitempty"`
	ServiceTier          string            `json:"service_tier,omitempty"`
	User                 string            `json:"user,omitempty"`
	PromptCacheKey       string            `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string            `json:"prompt_cache_retention,omitempty"`
}

// InputMessage represents a message input item.
type InputMessage struct {
	Type    string      `json:"type,omitempty"` // "message" (may be omitted for EasyInputMessage)
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string | []ContentPart
}

// InputFunctionCall represents a function_call input item.
type InputFunctionCall struct {
	Type      string `json:"type"` // "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// InputFunctionCallOutput represents a function_call_output input item.
type InputFunctionCallOutput struct {
	Type   string `json:"type"` // "function_call_output"
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

// ===== Content Part Types =====

// ContentPart represents a content item within an input message.
// Type determines which fields are populated.
type ContentPart struct {
	Type string `json:"type"` // "input_text", "input_image", "input_audio", "input_file"

	// input_text
	Text string `json:"text,omitempty"`

	// input_image
	ImageURL interface{} `json:"image_url,omitempty"` // string or {url, detail}
	FileID   string      `json:"file_id,omitempty"`
	Detail   string      `json:"detail,omitempty"`

	// input_audio
	Data   string `json:"data,omitempty"`   // base64 audio
	Format string `json:"format,omitempty"` // audio format

	// input_file
	Filename string `json:"filename,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
}

// OutputContent represents a content item within an output message item.
// Type determines which fields are populated.
type OutputContent struct {
	Type string `json:"type"` // "output_text", "output_refusal", "refusal", "reasoning_text", "summary_text", "text"

	// output_text, reasoning_text, summary_text, text
	Text string `json:"text,omitempty"`

	// output_refusal / refusal
	Refusal string `json:"refusal,omitempty"`

	// output_text: annotation citations (present but may be empty)
	Annotations []Annotation `json:"annotations,omitempty"`

	// output_text: per-token log probabilities (optional, included when top_logprobs requested)
	Logprobs interface{} `json:"logprobs,omitempty"`
}

// MarshalJSON keeps output_text.annotations present even when the list is empty.
// The Responses API returns an empty array for output_text annotations rather than
// omitting the field, so preserve that on round-trip.
func (o OutputContent) MarshalJSON() ([]byte, error) {
	raw := map[string]interface{}{
		"type": o.Type,
	}
	if o.Type == "output_text" {
		// Always include "text" for output_text — omitting it causes Python SDK to
		// parse the attribute as None instead of "", triggering TypeError in .join().
		raw["text"] = o.Text
		if o.Annotations == nil {
			raw["annotations"] = []Annotation{}
		} else {
			raw["annotations"] = o.Annotations
		}
	} else {
		if o.Text != "" {
			raw["text"] = o.Text
		}
		if len(o.Annotations) > 0 {
			raw["annotations"] = o.Annotations
		}
	}
	if o.Refusal != "" {
		raw["refusal"] = o.Refusal
	}
	if o.Logprobs != nil {
		raw["logprobs"] = o.Logprobs
	}
	return json.Marshal(raw)
}

// ===== Annotation Types =====

// Annotation is a union of annotation types attached to output text spans.
// Type is one of "file_citation", "url_citation", "container_file_citation".
type Annotation struct {
	Type string `json:"type"`

	// file_citation, container_file_citation
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`

	// file_citation
	Index int `json:"index,omitempty"`

	// url_citation
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`

	// url_citation, container_file_citation
	StartIndex int `json:"start_index,omitempty"`
	EndIndex   int `json:"end_index,omitempty"`

	// container_file_citation
	ContainerID string `json:"container_id,omitempty"`
}

// ===== Tool Types =====

// Tool represents a tool available to the model (flat Responses API format).
// Type determines which additional fields are populated.
type Tool struct {
	Type string `json:"type"` // "function", "web_search_preview", "web_search_preview_2025_03_11",
	// "file_search", "computer_use_preview", "code_interpreter",
	// "image_generation", "mcp", "local_shell", "bash", "custom"

	// function
	Name        string      `json:"name,omitempty"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`

	// web_search_preview
	UserLocation      interface{} `json:"user_location,omitempty"`
	SearchContextSize string      `json:"search_context_size,omitempty"`

	// file_search
	VectorStoreIDs []string    `json:"vector_store_ids,omitempty"`
	MaxNumResults  *int        `json:"max_num_results,omitempty"`
	RankingOptions interface{} `json:"ranking_options,omitempty"`
	Filters        interface{} `json:"filters,omitempty"`

	// computer_use_preview
	Environment   string `json:"environment,omitempty"`
	DisplayWidth  *int   `json:"display_width,omitempty"`
	DisplayHeight *int   `json:"display_height,omitempty"`

	// code_interpreter
	Container interface{} `json:"container,omitempty"`

	// mcp
	ServerLabel     string      `json:"server_label,omitempty"`
	ServerURL       string      `json:"server_url,omitempty"`
	Headers         interface{} `json:"headers,omitempty"`
	AllowedTools    interface{} `json:"allowed_tools,omitempty"`
	RequireApproval interface{} `json:"require_approval,omitempty"`
}

// ===== Reasoning =====

// Reasoning represents reasoning configuration.
type Reasoning struct {
	Effort          string `json:"effort,omitempty"` // "low","medium","high","none"
	Summary         string `json:"summary,omitempty"`
	GenerateSummary string `json:"generate_summary,omitempty"` // deprecated alias for Summary
}

// ===== Text Config =====

// TextConfig represents text output configuration in a request.
type TextConfig struct {
	Format    interface{} `json:"format,omitempty"` // {type:"text"} | {type:"json_schema",...} | {type:"json_object"}
	Verbosity string      `json:"verbosity,omitempty"`
}

// ===== Usage Types =====

// Usage represents token usage in a Responses API response.
type Usage struct {
	InputTokens         int           `json:"input_tokens"`
	OutputTokens        int           `json:"output_tokens"`
	TotalTokens         int           `json:"total_tokens"`
	InputTokensDetails  InputDetails  `json:"input_tokens_details"`
	OutputTokensDetails OutputDetails `json:"output_tokens_details"`
}

// InputDetails represents a breakdown of input token usage.
type InputDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens,omitempty"` // extension: audio input tokens
}

// OutputDetails represents a breakdown of output token usage.
type OutputDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
	AudioTokens     int `json:"audio_tokens,omitempty"` // extension: audio output tokens
	ImageTokens     int `json:"image_tokens,omitempty"` // extension: generated image/video tokens
}

// ===== Incomplete Details =====

// IncompleteDetails holds the reason a response was not fully completed.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// ===== Output Item Types =====

// OutputItem represents any item in the output (or echoed input) array.
// The Type field acts as a discriminator; only fields relevant to that type are populated.
//
// Supported types:
//
//	message, function_call, function_call_output,
//	web_search_call, file_search_call, image_generation_call,
//	code_interpreter_call, computer_call, computer_call_output,
//	reasoning, compaction,
//	local_shell_call, local_shell_call_output,
//	function_shell_call, function_shell_call_output,
//	apply_patch_tool_call, apply_patch_tool_call_output,
//	mcp_list_tools, mcp_approval_request, mcp_approval_response, mcp_tool_call,
//	custom_tool_call, custom_tool_call_output
type OutputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id,omitempty"`
	Status string `json:"status,omitempty"`

	// message: role, content, phase
	Role    string          `json:"role,omitempty"`
	Content []OutputContent `json:"content,omitempty"`
	Phase   string          `json:"phase,omitempty"` // "commentary" | "final_answer"

	// reasoning: content (reasoning text) + summary + encrypted_content
	Summary          []OutputContent `json:"summary,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`

	// function_call, computer_call, custom_tool_call: call_id + name + arguments
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// actor that created the item (function_call, web_search_call, image_generation_call, etc.)
	CreatedBy string `json:"created_by,omitempty"`

	// function_call_output, custom_tool_call_output, local_shell_call_output
	Output interface{} `json:"output,omitempty"`

	// web_search_call, computer_call, local_shell_call, function_shell_call: action (discriminated)
	Action interface{} `json:"action,omitempty"`

	// code_interpreter_call
	ContainerID string      `json:"container_id,omitempty"`
	Code        interface{} `json:"code,omitempty"`    // string | null
	Outputs     interface{} `json:"outputs,omitempty"` // []CodeInterpreterOutput | null

	// file_search_call
	Queries []string    `json:"queries,omitempty"`
	Results interface{} `json:"results,omitempty"` // []FileSearchResult | null

	// image_generation_call
	Result          string `json:"result,omitempty"` // base64-encoded image
	RevisedPrompt   string `json:"revised_prompt,omitempty"`
	ImageSize       string `json:"size,omitempty"`
	ImageQuality    string `json:"quality,omitempty"`
	ImageBackground string `json:"background,omitempty"`
	OutputFormat    string `json:"output_format,omitempty"`

	// computer_call: pending safety checks before execution
	PendingSafetyChecks []interface{} `json:"pending_safety_checks,omitempty"`

	// computer_call_output: screenshot URL after action
	CurrentURL string `json:"current_url,omitempty"`

	// mcp_tool_call
	ServerLabel string      `json:"server_label,omitempty"`
	Error       interface{} `json:"error,omitempty"` // MCPError | null

	// mcp_approval_request / mcp_approval_response
	ApprovalRequestID string `json:"approval_request_id,omitempty"`

	// apply_patch_tool_call: patch operation (discriminated: create/delete/update file)
	Operation interface{} `json:"operation,omitempty"`

	// custom_tool_call: JSON input to the custom tool
	Input interface{} `json:"input,omitempty"`
}

// ===== Response Types =====

// Response represents an OpenAI Responses API response object.
type Response struct {
	// Always-present identifier fields
	ID        string `json:"id"`
	Object    string `json:"object"` // "response"
	CreatedAt int64  `json:"created_at"`
	Model     string `json:"model"`
	Status    string `json:"status"` // "completed","failed","incomplete","in_progress","cancelled","queued"

	// Nullable required fields
	CompletedAt        *int64             `json:"completed_at"`
	IncompleteDetails  *IncompleteDetails `json:"incomplete_details"`
	PreviousResponseID interface{}        `json:"previous_response_id"` // null | string
	Instructions       interface{}        `json:"instructions"`         // null | string
	Error              interface{}        `json:"error"`                // null | error object
	Usage              *Usage             `json:"usage"`
	MaxOutputTokens    *int               `json:"max_output_tokens"`
	Reasoning          interface{}        `json:"reasoning"` // null | {effort, summary}
	User               interface{}        `json:"user,omitempty"`
	SafetyIdentifier   interface{}        `json:"safety_identifier"`
	PromptCacheKey     interface{}        `json:"prompt_cache_key"`

	// Required array / object fields (never null, may be empty)
	Output   []OutputItem      `json:"output"`
	Tools    []Tool            `json:"tools"`
	Metadata map[string]string `json:"metadata"`

	// Required boolean fields
	ParallelToolCalls bool `json:"parallel_tool_calls"`
	Store             bool `json:"store"`
	Background        bool `json:"background"`

	// Required numeric fields (schema requires them even when zero)
	PresencePenalty  float64 `json:"presence_penalty"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	TopLogprobs      int     `json:"top_logprobs"`

	// Required nullable numeric fields
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`

	// Required string fields (schema requires presence)
	Truncation  string `json:"truncation,omitempty"`
	ServiceTier string `json:"service_tier,omitempty"`

	// Tool choice (always present in real responses)
	ToolChoice interface{} `json:"tool_choice,omitempty"`

	// Text config echoed from request
	Text interface{} `json:"text,omitempty"` // {format: {type: "text"|"json_schema"|"json_object"}}

	// Required-nullable fields (must be present in JSON even when null)
	MaxToolCalls *int `json:"max_tool_calls"`

	// Input items echo (populated when include=["input_items"] is requested)
	Input []OutputItem `json:"input,omitempty"`

	// Follow-up response IDs (populated when include=["next_response_ids"] is requested)
	NextResponseIDs []string `json:"next_response_ids,omitempty"`

	// Billing / cache / conversation metadata
	CostToken            string      `json:"cost_token,omitempty"`
	Conversation         interface{} `json:"conversation,omitempty"`
	ContextEdits         interface{} `json:"context_edits,omitempty"`
	PromptCacheRetention interface{} `json:"prompt_cache_retention,omitempty"`
	Billing              interface{} `json:"billing,omitempty"`
}

// CompactResource represents the response from POST /v1/responses/compact.
type CompactResource struct {
	ID        string       `json:"id"`
	Object    string       `json:"object"` // "response.compaction"
	Output    []OutputItem `json:"output"`
	CreatedAt int64        `json:"created_at"`
	Usage     *Usage       `json:"usage"`
}
