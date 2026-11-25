package converterutil

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrImageBudgetExhausted is returned when a request asks for more images
// than the per-request budget allows.
var ErrImageBudgetExhausted = errors.New("image budget exhausted")

// ImageFetchBudget bounds the image downloads performed while converting one
// client request.
//
// A request may carry several image_url parts, and each one is fetched
// separately. Without a shared budget the added latency is the sum of the
// per-fetch timeouts — five images cost up to five times the timeout — and
// the router keeps downloading long after the client has given up or
// disconnected.
//
// The budget has two limits:
//
//   - a deadline shared by every fetch of the request, so the total time
//     spent downloading is bounded no matter how many images there are;
//   - a cap on the number of images, so a request carrying hundreds of
//     them cannot turn the router into a downloader.
//
// A nil *ImageFetchBudget means "no budget": every fetch runs with its own
// timeout. The methods handle a nil receiver, so callers do not have to
// check.
type ImageFetchBudget struct {
	mu        sync.Mutex
	deadline  time.Time
	maxImages int
	fetched   int
}

// NewImageFetchBudget returns a budget for one request. maxImages caps the
// number of downloads; timeout is the total time the request's downloads may
// take together. A non-positive maxImages means no cap.
func NewImageFetchBudget(maxImages int, timeout time.Duration) *ImageFetchBudget {
	budget := &ImageFetchBudget{maxImages: maxImages}
	if timeout > 0 {
		budget.deadline = time.Now().Add(timeout)
	}
	return budget
}

// Reserve accounts for one more image download.
//
// It reports false when the budget's image cap is already reached, in which
// case the caller must not download anything: the request has asked for
// more attachments than the router is willing to fetch.
func (b *ImageFetchBudget) Reserve() bool {
	if b == nil {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.maxImages > 0 && b.fetched >= b.maxImages {
		return false
	}
	b.fetched++
	return true
}

// Fetched returns how many image downloads the budget has accounted for.
func (b *ImageFetchBudget) Fetched() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.fetched
}

// Remaining reports how much of the budget's time is left.
//
// A negative result means the deadline has passed; zero means the budget
// has no time limit (or none left to grant). The caller passes the result
// to WithImageFetchDeadline to cut a fetch short.
func (b *ImageFetchBudget) Remaining() time.Duration {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.deadline.IsZero() {
		return 0
	}
	return time.Until(b.deadline)
}

// WithImageFetchDeadline returns a context that is cancelled when the
// budget's deadline passes.
//
// The per-fetch timeout still applies on top of this: the deadline is the
// request-wide budget, the timeout is the single fetch's own limit. The
// shorter of the two wins.
//
// A nil budget or a budget without a deadline yields the parent context
// unchanged.
func (b *ImageFetchBudget) WithImageFetchDeadline(parent context.Context) (context.Context, context.CancelFunc) {
	if b == nil {
		return parent, func() {}
	}
	b.mu.Lock()
	deadline := b.deadline
	b.mu.Unlock()
	if deadline.IsZero() {
		return parent, func() {}
	}
	return context.WithDeadline(parent, deadline)
}

// ConversionOptions carries the per-request settings a converter needs.
//
// It is passed by pointer through the converter call chain so that a
// request's image budget reaches the code that actually downloads the
// images.
type ConversionOptions struct {
	// ImageBudget bounds the image downloads for this request. A nil budget
	// means every download runs with its own timeout only.
	ImageBudget *ImageFetchBudget
}

// ConversionOption mutates ConversionOptions.
type ConversionOption func(*ConversionOptions)

// WithImageBudget returns an option that bounds this request's image
// downloads.
func WithImageBudget(budget *ImageFetchBudget) ConversionOption {
	return func(o *ConversionOptions) {
		o.ImageBudget = budget
	}
}

// ApplyConversionOptions folds the options into a ConversionOptions.
func ApplyConversionOptions(opts []ConversionOption) ConversionOptions {
	var options ConversionOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&options)
		}
	}
	return options
}
