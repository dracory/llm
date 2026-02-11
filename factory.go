package llm

import (
	"fmt"
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

	if provider == ProviderOpenAI && options.ApiKey == "" {
		return nil, fmt.Errorf("openai api key is required")
	}

	if provider == ProviderGemini && options.ApiKey == "" {
		return nil, fmt.Errorf("google gemini api key is required")
	}

	if provider == ProviderVertex && options.ProjectID == "" {
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

	if options.Temperature == nil {
		options.Temperature = PtrFloat64(0.7)
	}

	if options.Region == "" && provider == ProviderVertex {
		options.Region = "europe-west1"
	}

	return NewLLM(options)
}
