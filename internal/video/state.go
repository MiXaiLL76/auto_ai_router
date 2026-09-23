package video

var transitions = map[State]map[State]bool{
	StateReserving:       {StateReserving: true, StateQueued: true, StateCancelRequested: true, StateFailed: true},
	StateQueued:          {StateReadyToSubmit: true, StateCancelRequested: true, StateFailed: true},
	StateReadyToSubmit:   {StateSubmitting: true, StateCancelRequested: true, StateFailed: true},
	StateSubmitting:      {StateSubmitting: true, StateSubmitted: true, StateSubmissionUnknown: true, StateReleasing: true, StateCancelRequested: true},
	StateSubmitted:       {StateProcessing: true, StateStoring: true, StateReleasing: true, StateCancelRequested: true},
	StateProcessing:      {StateProcessing: true, StateStoring: true, StateReleasing: true, StateCancelRequested: true},
	StateStoring:         {StateStoring: true, StateSettling: true, StateReleasing: true},
	StateSettling:        {StateSettling: true, StateCompleted: true},
	StateCancelRequested: {StateCancelling: true, StateReleasing: true, StateStoring: true},
	StateCancelling:      {StateCancelling: true, StateReleasing: true, StateStoring: true},
	StateReleasing:       {StateReleasing: true, StateFailed: true, StateCancelled: true},
}

func CanTransition(from, to State) bool { return transitions[from][to] }
