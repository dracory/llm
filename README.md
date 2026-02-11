# LLM Package

[![Go Tests](https://github.com/dracory/llm/actions/workflows/tests.yml/badge.svg)](https://github.com/dracory/llm/actions/workflows/tests.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/dracory/llm.svg)](https://pkg.go.dev/github.com/dracory/llm)

A unified Go library for integrating with multiple LLM providers through a single, consistent interface.

## Overview

This package offers a provider-agnostic interface for interacting with LLM services:

- **OpenAI** — GPT-4.1, GPT-5.x, O4 models
- **Google Gemini** — Gemini 2.5 / 3.x models via the Gemini API
- **Google Vertex AI** — Gemini models on Google Cloud
- **Anthropic** — Claude Sonnet 4, Opus 4.x, Haiku 4.5
- **OpenRouter** — Access 50+ models from OpenAI, Anthropic, Google, Mistral, Qwen, xAI, DeepSeek, and more through a single API
- **Custom** — Any OpenAI-compatible endpoint
- **Mock** — For testing without API calls

## Installation

```bash
go get github.com/dracory/llm
```

Requires **Go 1.25** or later.

## Key Features

- **Provider Agnostic**: Use any supported LLM provider through a consistent interface
- **Extensible**: Add new providers by implementing `LlmInterface` or use `RegisterCustomProvider`
- **Multiple Output Formats**: Text, JSON, XML, YAML, and image generation (PNG/JPEG)
- **Embedding Support**: Generate text embeddings across providers
- **Configurable**: Fine-tune model parameters like temperature, token limits, and provider-specific options
- **Structured Logging**: Optional `slog.Logger` support for production observability
- **Thread-Safe**: Concurrent provider registration is protected by `sync.RWMutex`

## Quick Start

```go
package main

import (
    "fmt"
    "os"

    "github.com/dracory/llm"
)

func main() {
    // Create a text model
    engine, err := llm.TextModel(llm.ProviderOpenAI, llm.LlmOptions{
        ApiKey: os.Getenv("OPENAI_API_KEY"),
        Model:  "gpt-4.1-nano",
    })
    if err != nil {
        panic(err)
    }

    response, err := engine.GenerateText(
        "You are a helpful assistant.",
        "What is a contract?",
    )
    if err != nil {
        panic(err)
    }

    fmt.Println(response)
}
```

## Usage Examples

### Text Generation

```go
engine, err := llm.TextModel(llm.ProviderOpenAI, llm.LlmOptions{
    ApiKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4.1-nano",
})

response, err := engine.GenerateText(
    "You are a helpful assistant.",
    "Explain quantum computing in simple terms.",
)
```

### JSON Generation

```go
engine, err := llm.JSONModel(llm.ProviderGemini, llm.LlmOptions{
    ApiKey:      os.Getenv("GEMINI_API_KEY"),
    Model:       "gemini-2.5-flash",
    Temperature: 0.3,
})

jsonResponse, err := engine.GenerateJSON(
    "You are a data extraction assistant.",
    "Extract the name, age, and city from: John is 30 years old and lives in NYC.",
)
```

### Image Generation

```go
engine, err := llm.ImageModel(llm.ProviderOpenRouter, llm.LlmOptions{
    ApiKey: os.Getenv("OPENROUTER_API_KEY"),
    Model:  llm.OPENROUTER_MODEL_GPT_5_IMAGE,
})

imageBytes, err := engine.GenerateImage("A sunset over a mountain lake")
```

### Embedding Generation

```go
engine, err := llm.TextModel(llm.ProviderOpenRouter, llm.LlmOptions{
    ApiKey: os.Getenv("OPENROUTER_API_KEY"),
    Model:  llm.OPENROUTER_MODEL_TEXT_EMBEDDING_3_SMALL,
})

embeddings, err := engine.GenerateEmbedding("The quick brown fox")
```

### Using OpenRouter with Pre-defined Model Constants

```go
engine, err := llm.TextModel(llm.ProviderOpenRouter, llm.LlmOptions{
    ApiKey: os.Getenv("OPENROUTER_API_KEY"),
    Model:  llm.OPENROUTER_MODEL_CLAUDE_SONNET_4_5,
})

response, err := engine.GenerateText(
    "You are a code reviewer.",
    "Review this function for bugs: ...",
)
```

### Vertex AI with Credentials

```go
engine, err := llm.TextModel(llm.ProviderVertex, llm.LlmOptions{
    ProjectID:   "my-gcp-project",
    Region:      "europe-west1",
    Model:       "gemini-2.5-flash",
    MaxTokens:   8192,
    Temperature: 0.7,
    ProviderOptions: map[string]any{
        "credentials_json": os.Getenv("VERTEXAI_CREDENTIALS_JSON"),
    },
})
```

### Custom OpenAI-Compatible Endpoint

```go
engine, err := llm.TextModel(llm.ProviderCustom, llm.LlmOptions{
    ApiKey: "your-api-key",
    Model:  "your-model",
    ProviderOptions: map[string]any{
        "url": "https://your-endpoint.com/v1/chat/completions",
    },
})
```

### Per-Call Option Overrides

```go
// Override options on a per-call basis
response, err := engine.GenerateText(
    "You are a helpful assistant.",
    "Summarize this document: ...",
    llm.LlmOptions{
        MaxTokens:   2000,
        Temperature: 0.2,
    },
)
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

    // GenerateEmbedding generates embeddings for the given text
    GenerateEmbedding(text string) ([]float32, error)

    // Generate generates content (DEPRECATED: use GenerateText or GenerateJSON)
    Generate(systemPrompt string, userMessage string, options ...LlmOptions) (string, error)
}
```

## Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `Provider` | `Provider` | LLM provider to use (`openai`, `gemini`, `vertex`, `anthropic`, `openrouter`, `custom`, `mock`) |
| `ApiKey` | `string` | API key for the provider |
| `ProjectID` | `string` | GCP project ID (Vertex AI) |
| `Region` | `string` | GCP region (Vertex AI, defaults to `europe-west1`) |
| `Model` | `string` | Model identifier |
| `MaxTokens` | `int` | Maximum tokens to generate (default: 4096, Vertex: 8192) |
| `Temperature` | `float64` | Randomness control, 0.0–1.0 (default: 0.7) |
| `Verbose` | `bool` | Enable verbose logging |
| `Logger` | `*slog.Logger` | Structured logger for production use |
| `OutputFormat` | `OutputFormat` | Output format (`text`, `json`, `xml`, `yaml`, `image/png`, `image/jpeg`) |
| `ProviderOptions` | `map[string]any` | Provider-specific options (credentials, endpoint URLs, etc.) |
| `MockResponse` | `string` | Canned response for mock provider (excluded from JSON serialization) |

## Factory Functions

| Function | Description |
|----------|-------------|
| `TextModel(provider, options)` | Creates an LLM configured for text output |
| `JSONModel(provider, options)` | Creates an LLM configured for JSON output |
| `ImageModel(provider, options)` | Creates an LLM configured for image generation |
| `NewLLM(options)` | Low-level constructor with full control |

## OpenRouter Model Constants

The package provides pre-defined constants for popular models available via OpenRouter:

| Category | Examples |
|----------|----------|
| **OpenAI** | `OPENROUTER_MODEL_GPT_5_2`, `OPENROUTER_MODEL_GPT_5_2_CODEX`, `OPENROUTER_MODEL_O4_MINI` |
| **Anthropic** | `OPENROUTER_MODEL_CLAUDE_SONNET_4_5`, `OPENROUTER_MODEL_CLAUDE_OPUS_4_6`, `OPENROUTER_MODEL_CLAUDE_HAIKU_4_5` |
| **Google** | `OPENROUTER_MODEL_GEMINI_2_5_PRO`, `OPENROUTER_MODEL_GEMINI_3_PRO_PREVIEW` |
| **Mistral** | `OPENROUTER_MODEL_MISTRAL_MEDIUM_3_1`, `OPENROUTER_MODEL_DEVSTRAL_2512` |
| **Qwen** | `OPENROUTER_MODEL_QWEN_3_MAX_THINKING`, `OPENROUTER_MODEL_QWEN_3_CODER_NEXT` |
| **xAI** | `OPENROUTER_MODEL_GROK_3`, `OPENROUTER_MODEL_GROK_4` |
| **DeepSeek** | `OPENROUTER_MODEL_DEEPSEEK_V3_1` |
| **Image** | `OPENROUTER_MODEL_GPT_5_IMAGE`, `OPENROUTER_MODEL_GEMINI_2_5_FLASH_IMAGE` |
| **Embedding** | `OPENROUTER_MODEL_TEXT_EMBEDDING_3_LARGE`, `OPENROUTER_MODEL_QWEN_3_EMBEDDING_0_6B` |

See `openrouter_models.go` for the full list with pricing and context window sizes.

## Adding a Custom Provider

### Option 1: Use `RegisterCustomProvider`

```go
llm.RegisterCustomProvider("my-provider", func(options llm.LlmOptions) (llm.LlmInterface, error) {
    return NewMyProvider(options)
})

engine, err := llm.NewLLM(llm.LlmOptions{
    Provider: llm.Provider("my-provider"),
    ApiKey:   "...",
})
```

### Option 2: Implement `LlmInterface`

1. Create a new file `yourprovider_implementation.go`
2. Implement all methods of `LlmInterface`
3. Register via `RegisterProvider` in an `init()` function

```go
type myProvider struct {
    options llm.LlmOptions
}

func (p *myProvider) GenerateText(systemPrompt, userPrompt string, opts ...llm.LlmOptions) (string, error) {
    // Your implementation
}

func (p *myProvider) GenerateJSON(systemPrompt, userPrompt string, opts ...llm.LlmOptions) (string, error) {
    // Your implementation
}

func (p *myProvider) GenerateImage(prompt string, opts ...llm.LlmOptions) ([]byte, error) {
    // Your implementation
}

func (p *myProvider) GenerateEmbedding(text string) ([]float32, error) {
    // Your implementation
}

func (p *myProvider) Generate(systemPrompt, userMessage string, opts ...llm.LlmOptions) (string, error) {
    // Your implementation
}
```

## Provider-Specific Notes

### OpenAI
- Requires `OPENAI_API_KEY` environment variable or `ApiKey` option
- Image generation returns decoded PNG bytes via the DALL-E API
- Supports model and size overrides via options for image generation

### Gemini
- Requires `GEMINI_API_KEY` environment variable or `ApiKey` option
- Uses the `google.golang.org/genai` SDK with system instruction support
- Defaults to `gemini-2.5-flash` if no model is specified

### Vertex AI
- Requires GCP project ID and region
- Credentials can be supplied in several ways:
  1. `ProviderOptions["credentials_json"]` — raw service-account JSON string or `[]byte`
  2. `ProviderOptions["credentials_file"]` — path to a service-account JSON file
  3. Environment variables: `VERTEXAI_CREDENTIALS_JSON`, `VERTEXAI_CREDENTIALS_FILE`, or `GOOGLE_APPLICATION_CREDENTIALS`
  4. Application Default Credentials as fallback

### Anthropic
- Requires `ANTHROPIC_API_KEY` environment variable or `ApiKey` option
- Supports custom TLS configuration via provider options:
  - `anthropic_root_ca_file` / `ANTHROPIC_ROOT_CA_FILE` — custom root CA file
  - `anthropic_root_ca_pem` / `ANTHROPIC_ROOT_CA_PEM` — custom root CA PEM
  - `anthropic_spki_hash` / `ANTHROPIC_EXPECTED_SPKI_HASH` — certificate SPKI pin

### OpenRouter
- Requires `OPENROUTER_API_KEY` environment variable or `ApiKey` option
- Provides access to models from multiple providers through a single API
- Image generation uses the chat completions endpoint with `modalities: ["image", "text"]`
- Supports structured logging via `Logger` option

### Custom
- Requires an endpoint URL via `ProviderOptions["url"]`, `ProviderOptions["endpoint_url"]`, or `ProviderOptions["base_url"]`
- Sends OpenAI-compatible chat completion requests
- Falls back to plain-text response parsing if JSON parsing fails

## Testing

The package includes a mock implementation for testing:

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
```

The mock returns the first non-empty `MockResponse` it finds, checking in order:
1. Options passed to the specific method call
2. Options used when creating the LLM instance

### Running Tests

```bash
go test ./...
```

Integration tests are skipped automatically when API keys are not set.

## Utility Functions

- **`CountTokens(text string) int`** — Approximate token count (words + punctuation)
- **`EstimateMaxTokens(promptTokens, contextWindowSize int) int`** — Estimate remaining tokens in context window

## Best Practices

1. **Error Handling**: Always check for errors when calling LLM methods
2. **Environment Variables**: Store API keys in environment variables, not in code
3. **Prompt Engineering**: Craft clear system prompts for better results
4. **Token Management**: Be mindful of token limits for large inputs
5. **Structured Logging**: Use the `Logger` option for production observability
6. **Provider Fallback**: Implement fallbacks to handle provider outages

## Similar Projects

- [go-openai](https://github.com/sashabaranov/go-openai) — OpenAI Go client
- [gollm](https://github.com/teilomillet/gollm) — Go LLM library
- [fantasy](https://github.com/charmbracelet/fantasy) — Charmbracelet LLM toolkit
