package llm

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/samber/lo"
	"github.com/sashabaranov/go-openai"
)

// openrouterImplementation implements LlmInterface using OpenRouter (OpenAI-compatible API)
type openrouterImplementation struct {
	client      *openai.Client
	model       string
	maxTokens   int
	temperature float64
	verbose     bool
}

// newOpenRouterImplementation creates a new OpenRouter provider implementation
func newOpenRouterImplementation(options LlmOptions) (LlmInterface, error) {
	o := options

	apiKey := o.ApiKey
	if apiKey == "" {
		return nil, fmt.Errorf("OpenRouter API key is required")
	}

	model := o.Model
	if model == "" {
		// Default to a widely available OpenRouter model alias if not supplied
		model = "openrouter/auto"
	}

	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = "https://openrouter.ai/api/v1"

	return &openrouterImplementation{
		client:      openai.NewClientWithConfig(cfg),
		model:       model,
		maxTokens:   o.MaxTokens,
		temperature: o.Temperature,
		verbose:     o.Verbose,
	}, nil
}

// Generate implements LlmInterface
func (o *openrouterImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	ctx := context.Background()

	// Apply options if provided
	model := o.model
	if options.Model != "" {
		model = options.Model
	}

	maxTokens := o.maxTokens
	if options.MaxTokens > 0 {
		maxTokens = options.MaxTokens
	}

	temperature := o.temperature
	if options.Temperature > 0 {
		temperature = options.Temperature
	}

	// Configure response format based on output format
	responseFormat := &openai.ChatCompletionResponseFormat{}
	if options.OutputFormat == OutputFormatJSON {
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeJSONObject
	} else {
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeText
	}

	if o.verbose {
		fmt.Printf("OpenRouter request: model=%s, maxTokens=%d, temperature=%f\n", model, maxTokens, temperature)
		fmt.Printf("Response format: %v\n", responseFormat)
		fmt.Printf("System prompt: %s\n", systemPrompt)
		fmt.Printf("User prompt: %s\n", userMessage)
	}

	// Create request
	req := openai.ChatCompletionRequest{
		Model:          model,
		ResponseFormat: responseFormat,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
			{Role: openai.ChatMessageRoleUser, Content: userMessage},
		},
		MaxTokens:   maxTokens,
		Temperature: float32(temperature),
	}

	// Generate response
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if o.verbose {
			fmt.Printf("OpenRouter generation error: %v\n", err)
		}
		return "", err
	}

	if len(resp.Choices) == 0 {
		if o.verbose {
			fmt.Printf("no response from OpenRouter")
		}
		return "", fmt.Errorf("no response from OpenRouter")
	}

	response := resp.Choices[0].Message.Content
	if o.verbose {
		fmt.Printf("OpenRouter response: %s\n", response)
	}
	return strings.TrimSpace(response), nil
}

// GenerateText implements LlmInterface
func (o *openrouterImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return o.Generate(systemPrompt, userPrompt, options)
}

// GenerateJSON implements LlmInterface
func (o *openrouterImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return o.Generate(systemPrompt, userPrompt, options)
}

// GenerateImage implements LlmInterface
func (o *openrouterImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	ctx := context.Background()

	// Apply options if provided
	model := o.model
	if options.Model != "" {
		model = options.Model
	}

	// Default to DALL-E 3 if no model specified or using auto
	if model == "" || model == "openrouter/auto" {
		model = "google/gemini-2.5-flash-image"
	}

	// Default size
	size := openai.CreateImageSize1024x1024
	if s, ok := options.ProviderOptions["size"].(string); ok {
		switch s {
		case "256x256":
			size = openai.CreateImageSize256x256
		case "512x512":
			size = openai.CreateImageSize512x512
		case "1024x1024":
			size = openai.CreateImageSize1024x1024
		case "1792x1024":
			size = openai.CreateImageSize1792x1024
		case "1024x1792":
			size = openai.CreateImageSize1024x1792
		}
	}

	if o.verbose {
		fmt.Printf("OpenRouter image request: model=%s, size=%s, prompt=%s\n", model, size, prompt)
	}

	// Create image request
	req := openai.ImageRequest{
		Prompt:         prompt,
		Model:          model,
		Size:           size,
		ResponseFormat: openai.CreateImageResponseFormatB64JSON,
		N:              1,
	}

	// Generate image
	resp, err := o.client.CreateImage(ctx, req)
	if err != nil {
		if o.verbose {
			fmt.Printf("OpenRouter image generation error: %v\n", err)
		}
		return nil, err
	}

	if len(resp.Data) == 0 {
		if o.verbose {
			fmt.Printf("no image data in response")
		}
		return nil, fmt.Errorf("no image generated")
	}

	if resp.Data[0].B64JSON == "" {
		return nil, fmt.Errorf("no base64 data in response")
	}

	// Decode base64 to bytes
	data, err := base64.StdEncoding.DecodeString(resp.Data[0].B64JSON)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 image data: %v", err)
	}

	if o.verbose {
		fmt.Printf("Generated image of size %d bytes\n", len(data))
	}
	return data, nil
}
