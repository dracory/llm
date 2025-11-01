package llm

import (
	"fmt"

	"github.com/samber/lo"
)

// TextModel creates an LLM model for text output
func TextModel(provider Provider, options LlmOptions) (LlmInterface, error) {
	return createProvider(provider, OutputFormatText, options)
}

// JSONModel creates an LLM model for JSON output
func JSONModel(provider Provider, options LlmOptions) (LlmInterface, error) {
	return createProvider(provider, OutputFormatJSON, options)
}

// ImageModel creates an LLM model for image output
func ImageModel(provider Provider, options LlmOptions) (LlmInterface, error) {
	return createProvider(provider, OutputFormatImagePNG, options)
}

// createProvider is a convenience function to create an LLM provider instance with common configurations
func createProvider(provider Provider, outputFormat OutputFormat, options LlmOptions) (LlmInterface, error) {
	// Override provider and output format with the specified values
	options.Provider = provider
	options.OutputFormat = outputFormat

	if provider == ProviderGemini && options.ApiKey == "" {
		return nil, fmt.Errorf("google gemini api key is required")
	}

	if provider == ProviderVertex && options.ApiKey == "" {
		return nil, fmt.Errorf("vertexai project id is required")
	}

	if provider == ProviderAnthropic && options.ApiKey == "" {
		return nil, fmt.Errorf("anthropic api key is required")
	}

	if provider == ProviderOpenRouter && options.ApiKey == "" {
		return nil, fmt.Errorf("openrouter api key is required")
	}

	// Skip model check for mock provider
	if provider != ProviderMock && options.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	if options.MaxTokens == 0 {
		options.MaxTokens = 4096
		if provider == ProviderVertex {
			options.MaxTokens = 8192
		}
	}

	if options.Temperature == 0 {
		options.Temperature = 0.7
	}

	if options.ProjectID == "" && provider == ProviderVertex {
		return nil, fmt.Errorf("vertexai project id is required")
	}

	if options.Region == "" && provider == ProviderVertex {
		options.Region = "europe-west1"
	}

	supportedProviders := []Provider{
		ProviderOpenAI,
		ProviderGemini,
		ProviderVertex,
		ProviderAnthropic,
		ProviderMock,
		ProviderOpenRouter,
	}

	if !lo.Contains(supportedProviders, provider) {
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}

	return NewLLM(options)
}
