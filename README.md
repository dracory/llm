# LLM Package

The LLM (Language Learning Model) package provides a flexible, extensible interface for integrating with various AI language models in the RoastMyContract application.

## Overview

This package offers a unified interface for interacting with different LLM providers, including:

- OpenAI (GPT models)
- Google Gemini
- Google Vertex AI
- Anthropic (Claude models)
- Mock implementation for testing

The design follows the adapter pattern, allowing easy integration of new LLM providers without modifying existing code.

## Key Features

- **Provider Agnostic**: Use any supported LLM provider through a consistent interface
- **Extensible**: Easily add new LLM providers by implementing the `LlmInterface`
- **Configurable**: Fine-tune model parameters like temperature, token limits, etc.
- **Multiple Output Formats**: Support for text, JSON, and image generation
- **Fallback Mechanism**: Automatic fallback to alternative providers if the primary one fails

## Usage Examples

### Basic Usage

```go
// Create an LLM instance with a specific provider
llmEngine, err := llm.CreateProvider(llm.ProviderOpenAI, llm.OutputFormatText)
if err != nil {
    // Handle error
}

// Generate text
response, err := llmEngine.GenerateText(
    "You are a helpful assistant.", // System prompt
    "What is a contract?",          // User message
    llm.LlmOptions{},               // Use default options
)
```

### Using Helper Functions

```go
// Get an LLM engine with the default provider (from config)
llmEngine := helpers.GetLLMEngine(llm.OutputFormatJSON)

// Generate JSON
jsonResponse, err := llmEngine.GenerateJSON(
    "You are a legal assistant that analyzes contracts.",
    "Analyze this contract clause: ...",
    llm.LlmOptions{
        Temperature: 0.3, // Override default temperature
    },
)
```

### Custom Configuration

```go
// Create an LLM with custom options
llmEngine, err := llm.NewLLM(llm.LlmOptions{
    Provider:     llm.ProviderVertex,
    ProjectID:    "my-gcp-project",
    Region:       "europe-west1",
    Model:        "gemini-2.5-flash",
    MaxTokens:    8192,
    Temperature:  0.7,
    Verbose:      true,
    OutputFormat: llm.OutputFormatJSON,
})
```

## Adding a New Provider

To add a new LLM provider:

1. Create a new file `llm_yourprovider.go`
2. Implement the `LlmInterface` interface
3. Register your provider in the init function

Example:

```go
// Define your provider type
type YourProviderLLM struct {
    // Your provider-specific fields
}

// Implement the LlmInterface methods
func (y *YourProviderLLM) Generate(systemPrompt, userMessage string, options LlmOptions) (string, error) {
    // Your implementation
}

// Register your provider in init()
func init() {
    RegisterProvider("yourprovider", func(options LlmOptions) LlmInterface {
        return NewYourProviderLLM(options)
    })
}
```

## Interface

The core interface that all LLM providers must implement:

```go
type LlmInterface interface {
    // GenerateText generates a text response
    GenerateText(systemPrompt string, userPrompt string, options LlmOptions) (string, error)

    // GenerateJSON generates a JSON response
    GenerateJSON(systemPrompt string, userPrompt string, options LlmOptions) (string, error)

    // GenerateImage generates an image from a prompt
    GenerateImage(prompt string, options LlmOptions) ([]byte, error)

    // Generate is the core method for generating content
    Generate(systemPrompt string, userMessage string, options LlmOptions) (string, error)
}
```

## Configuration Options

The `LlmOptions` struct provides configuration options for LLM requests:

| Option | Description |
|--------|-------------|
| Provider | Which LLM provider to use |
| ApiKey | API key for the provider |
| ProjectID | Project ID (for Vertex AI) |
| Region | Region (for Vertex AI) |
| Model | Model name to use |
| MaxTokens | Maximum tokens to generate |
| Temperature | Controls randomness (0.0-1.0) |
| Verbose | Enable verbose logging |
| OutputFormat | Desired output format |
| ProviderOptions | Provider-specific options |

## Best Practices

1. **Error Handling**: Always check for errors when calling LLM methods
2. **Fallback Mechanism**: Implement fallbacks to handle provider outages
3. **Prompt Engineering**: Craft clear system prompts for better results
4. **Token Management**: Be mindful of token limits for large inputs
5. **Environment Variables**: Store API keys in environment variables, not in code

## Provider-Specific Notes

### OpenAI
- Supports GPT-3.5 and GPT-4 models
- Requires an OpenAI API key

### Gemini
- Supports Gemini Pro models
- Requires a Google API key

### Vertex AI
- Supports Gemini models on Google Cloud
- Requires GCP project ID and region
- Credentials stored in `vertexapicredentials.json`

### Anthropic
- Supports Claude models
- Requires an Anthropic API key

## Testing

The package includes a mock implementation for testing:

```go
// Create a mock LLM for testing
mockLLM := llm.NewMock()

// Use in tests
response, err := mockLLM.GenerateText("system", "user", llm.LlmOptions{})
```


## Similar Projects

- [llm](https://github.com/sashabaranov/go-openai)
- [gollm](https://github.com/teilomillet/gollm)
