package llm

import (
	"context"
	"strings"
)

// MockModel implements the ModelInterface for testing purposes
type MockModel struct {
	// Response is the predefined response to return
	Response CompletionResponse

	// Error is the predefined error to return
	Error error

	// Options for the mock model
	options ModelOptions
}

// NewMockModel creates a new mock model with a default response
func NewMockModel() *MockModel {
	return &MockModel{
		Response: CompletionResponse{
			Text:       "This is a mock response",
			TokensUsed: 5,
		},
		Error: nil,
		options: ModelOptions{
			Provider:     ProviderMock,
			OutputFormat: OutputFormatText,
			Model:        "mock-model",
			MaxTokens:    4096,
			Temperature:  0.7,
		},
	}
}

// NewMockModelWithOptions creates a new mock model with the specified options
func NewMockModelWithOptions(options ModelOptions) *MockModel {
	model := NewMockModel()
	model.options = options
	// Ensure the provider is set to mock
	model.options.Provider = ProviderMock
	return model
}

// Complete implements the ModelInterface
func (m *MockModel) Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error) {
	if m.Error != nil {
		return CompletionResponse{}, m.Error
	}

	// Validate request
	if request.UserPrompt == "" && request.SystemPrompt == "" {
		return CompletionResponse{}, ErrInvalidRequest
	}

	// Combine prompts for response
	prompt := request.UserPrompt
	if request.SystemPrompt != "" {
		if prompt != "" {
			prompt = request.SystemPrompt + "\n\n" + prompt
		} else {
			prompt = request.SystemPrompt
		}
	}

	// If custom response is empty, generate a simple echo response
	if m.Response.Text == "" {
		return CompletionResponse{
			Text:       "Echo: " + strings.TrimSpace(prompt),
			TokensUsed: len(strings.Fields(prompt)) + 1,
		}, nil
	}

	return m.Response, nil
}

// GetProvider implements the ModelInterface
func (m *MockModel) GetProvider() Provider {
	return ProviderMock
}

// GetOutputFormat implements the ModelInterface
func (m *MockModel) GetOutputFormat() OutputFormat {
	return m.options.OutputFormat
}

// GetApiKey implements the ModelInterface
func (m *MockModel) GetApiKey() string {
	return m.options.ApiKey
}

// GetModel implements the ModelInterface
func (m *MockModel) GetModel() string {
	return m.options.Model
}

// GetMaxTokens implements the ModelInterface
func (m *MockModel) GetMaxTokens() int {
	return m.options.MaxTokens
}

// GetTemperature implements the ModelInterface
func (m *MockModel) GetTemperature() float64 {
	return m.options.Temperature
}

// GetProjectID implements the ModelInterface
func (m *MockModel) GetProjectID() string {
	return m.options.ProjectID
}

// GetRegion implements the ModelInterface
func (m *MockModel) GetRegion() string {
	return m.options.Region
}

// GetVerbose implements the ModelInterface
func (m *MockModel) GetVerbose() bool {
	return m.options.Verbose
}
