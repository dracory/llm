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
llmEngine, err := llm.TextModel(llm.ProviderOpenAI, llm.LlmOptions{
    ApiKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4",
})
if err != nil {
    // Handle error
}

// Generate text
response, err := llmEngine.GenerateText(
    "You are a helpful assistant.", // System prompt
    "What is a contract?",          // User message
)
```

### Using Factory Functions

```go
// Create a JSON model
llmEngine, err := llm.JSONModel(llm.ProviderGemini, llm.LlmOptions{
    ApiKey:      os.Getenv("GEMINI_API_KEY"),
    Model:       "gemini-pro",
    Temperature: 0.3,
})

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
// Create an LLM with custom options using NewLLM directly
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

// Or use factory functions for convenience
llmEngine, err := llm.TextModel(llm.ProviderVertex, llm.LlmOptions{
    ProjectID:   "my-gcp-project",
    Region:      "europe-west1",
    Model:       "gemini-2.5-flash",
    MaxTokens:   8192,
    Temperature: 0.7,
})
```

## Adding a New Provider

To add a new LLM provider:

1. Create a new file `yourprovider_implementation.go`
2. Implement the `LlmInterface` interface
3. Register your provider in the init function

Example:

```go
// Define your provider type
type YourProviderLLM struct {
    options LlmOptions
    // Your provider-specific fields
}

// Implement the LlmInterface methods
func (y *YourProviderLLM) Generate(systemPrompt, userMessage string, options ...LlmOptions) (string, error) {
    // Merge options
    opts := y.options
    if len(options) > 0 {
        opts = mergeOptions(opts, options[0])
    }
    // Your implementation
}

// Register your provider in init()
func init() {
    RegisterProvider(ProviderYourProvider, func(options LlmOptions) (LlmInterface, error) {
        return newYourProviderImplementation(options)
    })
}
```

## Interface

The core interface that all LLM providers must implement:

```go
type LlmInterface interface {
    // GenerateText generates a text response
    GenerateText(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

    // GenerateJSON generates a JSON response
    GenerateJSON(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

    // GenerateImage generates an image from a prompt
    GenerateImage(prompt string, options ...LlmOptions) ([]byte, error)

    // Generate is the core method for generating content (DEPRECATED)
    Generate(systemPrompt string, userMessage string, options ...LlmOptions) (string, error)
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

## Factory Functions

The package provides convenient factory functions:

- **`TextModel(provider, options)`** - Creates an LLM configured for text output
- **`JSONModel(provider, options)`** - Creates an LLM configured for JSON output
- **`ImageModel(provider, options)`** - Creates an LLM configured for image generation

All factory functions require both a provider and options parameter:

```go
// Text model
textLLM, err := llm.TextModel(llm.ProviderOpenAI, llm.LlmOptions{
    ApiKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4",
})

// JSON model
jsonLLM, err := llm.JSONModel(llm.ProviderGemini, llm.LlmOptions{
    ApiKey: os.Getenv("GEMINI_API_KEY"),
    Model:  "gemini-pro",
})

// Image model
imageLLM, err := llm.ImageModel(llm.ProviderOpenAI, llm.LlmOptions{
    ApiKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "dall-e-3",
})
```

## Testing

The package includes a mock implementation for testing:

### Using Mock Responses in Tests

You can easily test your code by providing mock responses:

```go
// Create a mock LLM with a default response
mockLLM, _ := llm.NewLLM(llm.LlmOptions{
    Provider:     llm.ProviderMock,
    MockResponse: "This is a mock response",
})

// Or provide per-call mock responses
response, _ := mockLLM.GenerateText(
    "system prompt",
    "user message",
    llm.LlmOptions{
        MockResponse: "Specific response for this test case",
    },
)

// The mock will return the provided response
fmt.Println(response) // Output: Specific response for this test case
```

The mock will return the first non-empty MockResponse it finds, checking in this order:
1. The options passed to the specific function call
2. The options used when creating the LLM client
```


## Similar Projects

- [llm](https://github.com/sashabaranov/go-openai)
- [gollm](https://github.com/teilomillet/gollm)
