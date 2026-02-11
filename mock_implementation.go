package llm

import (
	"github.com/samber/lo"
)

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

// mockImplementation implements LlmInterface for Mock provider
type mockImplementation struct {
	options LlmOptions
}

// =======================================================================
// == IMPLEMENTATION
// =======================================================================

func (c *mockImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.FirstOr(opts, LlmOptions{})

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
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *mockImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *mockImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	//options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	//options.OutputFormat = OutputFormatImagePNG
	return nil, nil
}

func (m *mockImplementation) GenerateEmbedding(text string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}
