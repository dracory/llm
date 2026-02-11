package llm

import (
	"fmt"
	"log/slog"
	"sync"
)

// LlmInterface is an interface for making LLM API calls
type LlmInterface interface {
	// GenerateText generates a text response from the LLM based on the given prompt
	GenerateText(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

	// GenerateJSON generates a JSON response from the LLM based on the given prompt
	GenerateJSON(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

	// GenerateImage generates an image from the LLM based on the given prompt
	GenerateImage(prompt string, options ...LlmOptions) ([]byte, error)

	// DEPRECATED: Generate generates a response from the LLM based on the given prompt and options
	Generate(systemPrompt string, userMessage string, options ...LlmOptions) (string, error)

	// GenerateEmbedding generates embeddings for the given text
	GenerateEmbedding(text string) ([]float32, error)
}

type LlmOptions struct {
	// Provider specifies which LLM provider to use
	Provider Provider

	// MockResponse, if not empty, will be returned by the mock implementation
	// instead of making an actual API call. This is useful for testing.
	MockResponse string `json:"-"`

	// ApiKey specifies the API key for the LLM provider
	ApiKey string

	// ProjectID specifies the project ID for the LLM (used by Vertex AI)
	ProjectID string

	// Region specifies the region for the LLM (used by Vertex AI)
	Region string

	// Model specifies the LLM model to use
	Model string

	// MaxTokens specifies the maximum number of tokens to generate
	MaxTokens int

	// Temperature controls the randomness of the response
	// A higher temperature (e.g., 0.8) makes the output more random and creative,
	// while a lower temperature (e.g., 0.2) makes the output more focused and deterministic.
	Temperature float64

	// Verbose controls whether to log detailed information
	Verbose bool

	// Logger specifies a logger to use for error logging
	Logger *slog.Logger

	// OutputFormat specifies the output format from the LLM
	OutputFormat OutputFormat

	// Additional options specific to the LLM provider
	ProviderOptions map[string]any
}

// LlmFactory is a function type that creates a new LLM instance
// Now returns (LlmInterface, error)
type LlmFactory func(options LlmOptions) (LlmInterface, error)

var (
	// providerMu protects providerFactories from concurrent access
	providerMu sync.RWMutex
	// providerFactories maps provider names to their factory functions
	providerFactories = make(map[Provider]LlmFactory)
)

// RegisterProvider registers a new LLM provider factory
func RegisterProvider(provider Provider, factory LlmFactory) {
	providerMu.Lock()
	defer providerMu.Unlock()
	providerFactories[provider] = factory
}

// RegisterCustomProvider registers a custom LLM provider
func RegisterCustomProvider(name string, factory LlmFactory) {
	RegisterProvider(Provider(name), factory)
}

// NewLLM creates a new LLM instance based on the provider specified in options
func NewLLM(options LlmOptions) (LlmInterface, error) {
	if options.Provider == "" {
		// Default to OpenAI if no provider is specified
		options.Provider = ProviderOpenAI
	}

	providerMu.RLock()
	factory, exists := providerFactories[options.Provider]
	providerMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("unsupported LLM provider: %s", options.Provider)
	}

	llm, err := factory(options)
	if err != nil {
		return nil, err
	}
	return llm, nil
}

// init registers the built-in LLM providers
func init() {
	// Register built-in providers
	RegisterProvider(ProviderOpenAI, func(options LlmOptions) (LlmInterface, error) {
		return newOpenaiImplementation(options)
	})

	RegisterProvider(ProviderGemini, func(options LlmOptions) (LlmInterface, error) {
		return newGeminiImplementation(options)
	})

	RegisterProvider(ProviderVertex, func(options LlmOptions) (LlmInterface, error) {
		return newVertexImplementation(options)
	})

	RegisterProvider(ProviderMock, func(options LlmOptions) (LlmInterface, error) {
		return newMockImplementation(options)
	})

	RegisterProvider(ProviderAnthropic, func(options LlmOptions) (LlmInterface, error) {
		return newAnthropicImplementation(options)
	})

	RegisterProvider(ProviderOpenRouter, func(options LlmOptions) (LlmInterface, error) {
		return newOpenRouterImplementation(options)
	})

	RegisterProvider(ProviderCustom, func(options LlmOptions) (LlmInterface, error) {
		return newCustomImplementation(options)
	})
}
