# LLM Package

Go package for Large Language Model (LLM) operations.

## Installation

```bash
go get github.com/dracory/base/llm
```

## Usage

### Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"github.com/dracory/base/llm"
)

func main() {
	// Create a model with OpenAI provider for text output
	model, err := llm.TextModel(llm.ProviderOpenAI)
	if err != nil {
		fmt.Printf("Error creating model: %v\n", err)
		return
	}

	// Generate a completion
	response, err := model.Complete(context.Background(), llm.CompletionRequest{
		Prompt:      "Once upon a time",
		MaxTokens:   100,
		Temperature: 0.7,
	})
	if err != nil {
		fmt.Printf("Error generating completion: %v\n", err)
		return
	}

	fmt.Println("Generated text:", response.Text)
	fmt.Println("Tokens used:", response.TokensUsed)
}
```

### Creating Different Output Format Models

The package provides convenience functions for creating models with specific output formats:

```go
// For text output
textModel, err := llm.TextModel(llm.ProviderOpenAI)

// For JSON output
jsonModel, err := llm.JSONModel(llm.ProviderOpenAI)

// For image output
imageModel, err := llm.ImageModel(llm.ProviderOpenAI)
```

### Using Different Providers

The package supports multiple LLM providers:

```go
// OpenAI
openaiModel, err := llm.TextModel(llm.ProviderOpenAI)

// Google Gemini
geminiModel, err := llm.TextModel(llm.ProviderGemini)

// Google Vertex AI
vertexModel, err := llm.TextModel(llm.ProviderVertex)

// Anthropic (Claude)
anthropicModel, err := llm.TextModel(llm.ProviderAnthropic)

// Mock model for testing
mockModel, err := llm.TextModel(llm.ProviderMock)
```

### Advanced Configuration

You can create a model with custom configuration options:

```go
options := llm.ModelOptions{
	Provider:     llm.ProviderOpenAI,
	OutputFormat: llm.OutputFormatJSON,
	ApiKey:       "your-api-key",
	Model:        "gpt-4",
	MaxTokens:    2048,
	Temperature:  0.5,
	Verbose:      true,
}

model, err := llm.NewModel(options)
if err != nil {
	// Handle error
}
```

### ModelInterface

The package defines a `ModelInterface` that all models implement:

```go
type ModelInterface interface {
	// Complete generates a completion for the provided prompt
	Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error)
}
```

## Available Output Formats

The package supports the following output formats:

- `OutputFormatText` - Plain text output
- `OutputFormatJSON` - JSON formatted output
- `OutputFormatXML` - XML formatted output
- `OutputFormatYAML` - YAML formatted output
- `OutputFormatEnum` - Enumeration values
- `OutputFormatImagePNG` - PNG image output
- `OutputFormatImageJPG` - JPEG image output

## License

This package is part of the Dracory/Base project and is subject to the same licensing terms.
