package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/samber/lo"
	"github.com/sashabaranov/go-openai"
)

// openaiImplementation implements LlmInterface using OpenAI's API
type openaiImplementation struct {
	client      *openai.Client
	model       string
	maxTokens   int
	temperature float64
	verbose     bool
}

// newOpenaiImplementation creates a new OpenAI provider implementation
func newOpenaiImplementation(options LlmOptions) (LlmInterface, error) {
	o := options

	apiKey := o.ApiKey
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	model := o.Model
	if model == "" {
		model = openai.GPT4TurboPreview
	}

	return &openaiImplementation{
		client:      openai.NewClient(apiKey),
		model:       model,
		maxTokens:   o.MaxTokens,
		temperature: o.Temperature,
		verbose:     o.Verbose,
	}, nil
}

// Generate implements LlmInterface
func (o *openaiImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
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
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userMessage,
			},
		},
		MaxTokens:   maxTokens,
		Temperature: float32(temperature),
	}

	// Generate response
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if o.verbose {
			fmt.Printf("OpenAI generation error: %v\n", err)
		}
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from OpenAI")
	}

	response := resp.Choices[0].Message.Content
	return strings.TrimSpace(response), nil
}

// GenerateText implements LlmInterface
func (o *openaiImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return o.Generate(systemPrompt, userPrompt, options)
}

// GenerateJSON implements LlmInterface
func (o *openaiImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return o.Generate(systemPrompt, userPrompt, options)
}

// GenerateImage implements LlmInterface
func (o *openaiImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	_ = lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	ctx := context.Background()

	// Use DALL-E model for image generation
	req := openai.ImageRequest{
		Prompt: prompt,
		Size:   openai.CreateImageSize1024x1024,
		N:      1,
	}

	resp, err := o.client.CreateImage(ctx, req)
	if err != nil {
		if o.verbose {
			fmt.Printf("OpenAI image generation error: %v\n", err)
		}
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no image generated")
	}

	// Note: This returns the URL of the image, not the image bytes
	// To get actual image bytes, you would need to download from the URL
	return []byte(resp.Data[0].URL), nil
}
