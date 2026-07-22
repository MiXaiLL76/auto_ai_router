package kafkalog

import "errors"

// ErrQueueFull is returned when the spend event queue is full and the
// backpressure timeout is reached.
var ErrQueueFull = errors.New("kafkalog: spend log queue full - timeout reached")
