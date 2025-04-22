package llm

import (
	"context"
	"errors"
	"fmt"
)

// ErrInvalidRequest is returned when a request is invalid
var ErrInvalidRequest = errors.New("invalid request")

// ErrServiceUnavailable is returned when the LLM service is unavailable
var ErrServiceUnavailable = errors.New("service unavailable")

// config is a temporary placeholder for configuration
// This should be replaced with actual configuration access
var config struct {
	Debug                    bool
	OpenAiApiKey             string
	OpenAiDefaultModel       string
	GoogleGeminiApiKey       string
	GoogleGeminiDefaultModel string
	VertexAiProjectID        string
	VertexAiDefaultModel     string
	VertexAiRegion           string
	AnthropicApiKey          string
	AnthropicDefaultModel    string
}

// CompletionRequest represents a request to generate a completion
type CompletionRequest struct {
	// SystemPrompt contains instructions for the LLM
	SystemPrompt string `json:"system_prompt"`

	// UserPrompt contains the actual query or content to process
	UserPrompt string `json:"user_prompt"`

	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int `json:"max_tokens"`

	// Temperature controls randomness in generation (0.0-1.0)
	Temperature float64 `json:"temperature"`
}

// CompletionResponse represents a response from a completion request
type CompletionResponse struct {
	// Text is the generated completion text
	Text string `json:"text"`

	// TokensUsed is the number of tokens used for this request
	TokensUsed int `json:"tokens_used"`
}

// modelImplementation is the concrete implementation of ModelInterface
type modelImplementation struct {
	options ModelOptions
}

// Complete implements the ModelInterface
func (m *modelImplementation) Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error) {
	// This is a placeholder implementation
	// Actual implementation would need to handle different providers and output formats

	// For now, return a mock response
	if m.options.Provider == ProviderMock {
		return CompletionResponse{
			Text:       "This is a mock completion response",
			TokensUsed: 7,
		}, nil
	}

	// TODO: Implement provider-specific completion logic
	return CompletionResponse{}, fmt.Errorf("provider %s not yet implemented", m.options.Provider)
}

// GetProvider returns the provider of the model
func (m *modelImplementation) GetProvider() Provider {
	return m.options.Provider
}

// GetOutputFormat returns the output format of the model
func (m *modelImplementation) GetOutputFormat() OutputFormat {
	return m.options.OutputFormat
}

// GetApiKey returns the API key of the model
func (m *modelImplementation) GetApiKey() string {
	return m.options.ApiKey
}

// GetModel returns the model of the model
func (m *modelImplementation) GetModel() string {
	return m.options.Model
}

// GetMaxTokens returns the maximum number of tokens of the model
func (m *modelImplementation) GetMaxTokens() int {
	return m.options.MaxTokens
}

// GetTemperature returns the temperature of the model
func (m *modelImplementation) GetTemperature() float64 {
	return m.options.Temperature
}

// GetProjectID returns the project ID of the model
func (m *modelImplementation) GetProjectID() string {
	return m.options.ProjectID
}

// GetRegion returns the region of the model
func (m *modelImplementation) GetRegion() string {
	return m.options.Region
}

// GetVerbose returns the verbose of the model
func (m *modelImplementation) GetVerbose() bool {
	return m.options.Verbose
}
