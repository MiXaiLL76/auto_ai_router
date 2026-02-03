package transform

import "context"

// RequestTransformer converts OpenAI format requests to provider-specific format
type RequestTransformer interface {
	// TransformOpenAIRequest converts OpenAI format request to provider-specific format
	TransformOpenAIRequest(ctx context.Context, body []byte, opts ...RequestOption) ([]byte, error)
}

// ResponseTransformer converts provider-specific responses back to OpenAI format
type ResponseTransformer interface {
	// TransformProviderResponse converts provider-specific response to OpenAI format
	TransformProviderResponse(ctx context.Context, body []byte, opts ...ResponseOption) ([]byte, error)
}

// RequestOption allows configuring request transformation behavior
type RequestOption func(*requestConfig)

type requestConfig struct {
	ModelID            string
	IsImageGeneration  bool
	IsStreamingRequest bool
}

// ResponseOption allows configuring response transformation behavior
type ResponseOption func(*responseConfig)

type responseConfig struct {
	ModelID             string
	IsStreamingResponse bool
}

// WithModelID sets the model ID for transformation
func WithModelID(modelID string) interface{} {
	return func(c interface{}) {
		if rc, ok := c.(*requestConfig); ok {
			rc.ModelID = modelID
		} else if rsc, ok := c.(*responseConfig); ok {
			rsc.ModelID = modelID
		}
	}
}

// WithIsImageGeneration marks if request is for image generation
func WithIsImageGeneration(isImage bool) RequestOption {
	return func(c *requestConfig) {
		c.IsImageGeneration = isImage
	}
}

// WithIsStreamingRequest marks if request is streaming
func WithIsStreamingRequest(isStreaming bool) RequestOption {
	return func(c *requestConfig) {
		c.IsStreamingRequest = isStreaming
	}
}

// WithIsStreamingResponse marks if response is streaming
func WithIsStreamingResponse(isStreaming bool) ResponseOption {
	return func(c *responseConfig) {
		c.IsStreamingResponse = isStreaming
	}
}
