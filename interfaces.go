package llm

import "context"

// ModelInterface defines the interface for interacting with Large Language Models
type ModelInterface interface {
	// Complete generates a completion for the provided prompt
	Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error)

	// GetProvider returns the provider of the model
	GetProvider() Provider

	// GetOutputFormat returns the output format of the model
	GetOutputFormat() OutputFormat

	// GetApiKey returns the API key of the model
	GetApiKey() string

	// GetModel returns the model of the model
	GetModel() string

	// GetMaxTokens returns the maximum number of tokens of the model
	GetMaxTokens() int

	// GetTemperature returns the temperature of the model
	GetTemperature() float64

	// GetProjectID returns the project ID of the model
	GetProjectID() string

	// GetRegion returns the region of the model
	GetRegion() string

	// GetVerbose returns the verbose of the model
	GetVerbose() bool
}

// ModelOptions contains configuration options for creating an LLM model
type ModelOptions struct {
	Provider     Provider
	OutputFormat OutputFormat
	ApiKey       string
	Model        string
	MaxTokens    int
	Temperature  float64
	ProjectID    string
	Region       string
	Verbose      bool
}
