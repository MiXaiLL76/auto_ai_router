package worker

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
)

// Job represents a unit of work to be processed by a worker.
// Implementations should define their own concrete job types with this as a base.
type Job interface {
	// Execute performs the work synchronously.
	// Context should be used to check for cancellation.
	Execute(ctx context.Context) Result
}

// Result represents the outcome of a job execution.
type Result interface {
	// Error returns any error that occurred during execution, or nil if successful.
	Error() error
}

// SpawnWorkerPool creates and manages a pool of worker goroutines.
// Workers process jobs from the provided job queue and send results back through result channels.
//
// Parameters:
//   - ctx: Context for cancellation. Workers will exit when context is cancelled.
//   - numWorkers: Number of concurrent worker goroutines to spawn.
//   - jobQueue: Channel to receive jobs. Workers will read from this channel.
//   - logger: Logger for worker lifecycle and error logging.
//
// Returns:
//   - WaitGroup that tracks all worker goroutines. Call Wait() to block until all workers exit.
//
// Example:
//
//	type MyJob struct {
//	    name string
//	    result chan<- MyResult
//	}
//
//	type MyResult struct {
//	    err error
//	}
//
//	func (j MyJob) Execute(ctx context.Context) Result {
//	    // Do work
//	    return MyResult{err: nil}
//	}
//
//	jobQueue := make(chan Job, 10)
//	wg := SpawnWorkerPool(ctx, 5, jobQueue, logger)
//
//	// Send jobs
//	jobQueue <- myJob
//
//	// Wait for completion
//	wg.Wait()
func SpawnWorkerPool(
	ctx context.Context,
	numWorkers int,
	jobQueue <-chan Job,
	logger *slog.Logger,
) *sync.WaitGroup {
	if numWorkers <= 0 {
		numWorkers = 1
	}

	wg := &sync.WaitGroup{}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			logger.Debug("Worker started",
				"worker_id", workerID,
				"total_workers", numWorkers,
			)

			executeJob := func(job Job) {
				defer func() {
					if r := recover(); r != nil {
						logger.Error("Job panicked",
							"worker_id", workerID,
							"panic", fmt.Sprintf("%v", r),
						)
					}
				}()

				result := job.Execute(ctx)

				// Log any errors that occurred
				if result != nil && result.Error() != nil {
					logger.Error("Job execution failed",
						"worker_id", workerID,
						"error", result.Error(),
					)
				}
			}

			for {
				select {
				case <-ctx.Done():
					// Context cancelled, drain remaining buffered jobs before exiting
					logger.Debug("Worker draining remaining jobs",
						"worker_id", workerID,
						"reason", "context_cancelled",
					)
					for job := range jobQueue {
						executeJob(job)
					}
					logger.Debug("Worker exiting",
						"worker_id", workerID,
						"reason", "context_cancelled",
					)
					return

				case job, ok := <-jobQueue:
					if !ok {
						// Job queue closed, exit worker
						logger.Debug("Worker exiting",
							"worker_id", workerID,
							"reason", "job_queue_closed",
						)
						return
					}

					executeJob(job)
				}
			}
		}(i)
	}

	logger.Debug("Worker pool spawned",
		"num_workers", numWorkers,
	)

	return wg
}
