package llm

import "fmt"

// NewModel creates a new LLM model based on the provided options
func NewModel(options ModelOptions) (ModelInterface, error) {
	// Validate required options
	if options.Provider == "" {
		return nil, fmt.Errorf("provider is required")
	}

	// For specific provider implementations
	switch options.Provider {
	case ProviderGemini:
		return newGeminiModel(options)
	case ProviderOpenAI:
		return newOpenAIModel(options)
	case ProviderVertex:
		return newVertexModel(options)
	case ProviderMock:
		// For mock provider, use the MockModel with the provided options
		return NewMockModelWithOptions(options), nil
	}

	// For other providers, use the generic implementation
	return &modelImplementation{
		options: options,
	}, nil
}

// TextModel creates an LLM model for text output
func TextModel(provider Provider) (ModelInterface, error) {
	return createProvider(provider, OutputFormatText)
}

// JSONModel creates an LLM model for JSON output
func JSONModel(provider Provider) (ModelInterface, error) {
	return createProvider(provider, OutputFormatJSON)
}

// ImageModel creates an LLM model for image output
func ImageModel(provider Provider) (ModelInterface, error) {
	return createProvider(provider, OutputFormatImagePNG)
}

// createProvider is a convenience function to create an LLM provider instance with common configurations
func createProvider(provider Provider, outputFormat OutputFormat) (ModelInterface, error) {
	options := ModelOptions{
		Provider:     provider,
		OutputFormat: outputFormat,
		Verbose:      config.Debug,
	}

	if provider == ProviderGemini && config.GoogleGeminiApiKey == "" {
		return nil, fmt.Errorf("google Gemini API key is required")
	}

	if provider == ProviderVertex && config.VertexAiProjectID == "" {
		return nil, fmt.Errorf("vertex AI project ID is required")
	}

	if provider == ProviderAnthropic && config.AnthropicApiKey == "" {
		return nil, fmt.Errorf("anthropic API key is required")
	}

	if provider == ProviderOpenAI && config.OpenAiApiKey == "" {
		return nil, fmt.Errorf("openai API key is required")
	}

	// Apply provider-specific configurations
	switch provider {
	case ProviderOpenAI:
		options.ApiKey = config.OpenAiApiKey
		options.Model = config.OpenAiDefaultModel
		options.MaxTokens = 4096
		options.Temperature = 0.7
	case ProviderGemini:
		options.ApiKey = config.GoogleGeminiApiKey
		options.Model = config.GoogleGeminiDefaultModel
		options.MaxTokens = 4096
		options.Temperature = 0.7
	case ProviderVertex:
		options.ProjectID = config.VertexAiProjectID
		options.Region = config.VertexAiRegion
		if options.Region == "" {
			options.Region = "us-central1" // Default region
		}
		options.Model = config.VertexAiDefaultModel
		options.MaxTokens = 8192
		options.Temperature = 0.7
	case ProviderAnthropic:
		options.ApiKey = config.AnthropicApiKey
		options.Model = config.AnthropicDefaultModel
		options.MaxTokens = 4096
		options.Temperature = 0.7
	case ProviderMock:
		// No specific configuration needed for mock
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}

	return NewModel(options)
}
