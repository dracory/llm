package llm

import (
	"context"
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
		return "", fmt.Errorf("no response from OpenRouter")
	}

	response := resp.Choices[0].Message.Content
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
	_ = lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	return nil, fmt.Errorf("image generation is not supported via OpenRouter in this implementation")
}
