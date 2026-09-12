package llm

import "sync"

// =======================================================================
// == CONSTRUCTOR
// =======================================================================

func newMockImplementation(options LlmOptions) (LlmInterface, error) {
	// Set default model if not provided
	if options.Model == "" {
		options.Model = "mock-model"
	}
	return &mockImplementation{
		options: options,
	}, nil
}

// =======================================================================
// == TYPE
// =======================================================================

// mockImplementation implements LlmInterface for Mock provider.
//
// It also implements MockInterface, exposing Calls() and Reset() so tests
// can assert which calls were made (and in what order) and simulate API
// errors via LlmOptions.MockError.
type mockImplementation struct {
	options LlmOptions

	mu    sync.Mutex
	calls []MockCall
}

// =======================================================================
// == IMPLEMENTATION
// =======================================================================

// recordCall appends a call to the history. The options stored are the
// per-call options (or empty when none were supplied).
func (c *mockImplementation) recordCall(method, systemPrompt, userPrompt string, opts LlmOptions) {
	// Avoid retaining the caller's options map mutably by shallow copy.
	stored := opts
	c.mu.Lock()
	c.calls = append(c.calls, MockCall{
		Method:       method,
		SystemPrompt: systemPrompt,
		UserPrompt:   userPrompt,
		Options:      stored,
	})
	c.mu.Unlock()
}

func (c *mockImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := LlmOptions{}
	if len(opts) > 0 {
		options = opts[0]
	}

	c.recordCall("Generate", systemPrompt, userMessage, options)

	// MockError takes precedence over MockResponse.
	if options.MockError != nil {
		return "", options.MockError
	}
	if c.options.MockError != nil {
		return "", c.options.MockError
	}

	// Return mock response if provided in options
	if options.MockResponse != "" {
		return options.MockResponse, nil
	}

	// Or use the one from the client options
	if c.options.MockResponse != "" {
		return c.options.MockResponse, nil
	}

	// Handle empty input
	if userMessage == "" {
		return "", nil
	}

	return "", nil
}

func (c *mockImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatText
	return c.Generate(systemPrompt, userPrompt, perCall)
}

func (c *mockImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatJSON
	return c.Generate(systemPrompt, userPrompt, perCall)
}

func (c *mockImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	c.recordCall("GenerateImage", "", prompt, perCall)

	if perCall.MockError != nil {
		return nil, perCall.MockError
	}
	if c.options.MockError != nil {
		return nil, c.options.MockError
	}
	//options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	//options.OutputFormat = OutputFormatImagePNG
	return nil, nil
}

func (m *mockImplementation) GenerateEmbedding(text string) ([]float32, error) {
	m.recordCall("GenerateEmbedding", "", text, LlmOptions{})
	if m.options.MockError != nil {
		return nil, m.options.MockError
	}
	return []float32{0.1, 0.2, 0.3}, nil
}

// =======================================================================
// == MockInterface
// =======================================================================

// Calls returns a copy of all recorded calls in invocation order.
func (c *mockImplementation) Calls() []MockCall {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]MockCall, len(c.calls))
	copy(out, c.calls)
	return out
}

// Reset clears the recorded call history.
func (c *mockImplementation) Reset() {
	c.mu.Lock()
	c.calls = nil
	c.mu.Unlock()
}
