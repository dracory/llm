package llm

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// openaiImplementation implements LlmInterface using OpenAI's API
type openaiImplementation struct {
	client      *openai.Client
	model       string
	maxTokens   int
	temperature float64
	verbose     bool
	logger      *slog.Logger
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
		temperature: derefFloat64(o.Temperature, 0.7),
		verbose:     o.Verbose,
		logger:      o.Logger,
	}, nil
}

// baseOptions returns the base LlmOptions from the struct fields for merging.
func (o *openaiImplementation) baseOptions() LlmOptions {
	return LlmOptions{
		Model:       o.model,
		MaxTokens:   o.maxTokens,
		Temperature: &o.temperature,
		Verbose:     o.verbose,
		Logger:      o.logger,
	}
}

// Generate implements LlmInterface
func (o *openaiImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	merged := mergeOptions(o.baseOptions(), perCall)

	ctx := context.Background()

	model := merged.Model
	maxTokens := merged.MaxTokens
	temperature := derefFloat64(merged.Temperature, o.temperature)

	// Configure response format based on output format
	responseFormat := &openai.ChatCompletionResponseFormat{}
	if merged.OutputFormat == OutputFormatJSON {
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
		if o.logger != nil {
			o.logger.Error("OpenAI generation error",
				slog.String("error", err.Error()),
				slog.String("model", model))
		} else if o.verbose {
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
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatText
	return o.Generate(systemPrompt, userPrompt, perCall)
}

// GenerateJSON implements LlmInterface
func (o *openaiImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatJSON
	return o.Generate(systemPrompt, userPrompt, perCall)
}

// GenerateImage implements LlmInterface
func (o *openaiImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	merged := mergeOptions(o.baseOptions(), perCall)
	ctx := context.Background()

	model := merged.Model

	// Determine image size from provider options or default
	size := openai.CreateImageSize1024x1024
	if merged.ProviderOptions != nil {
		if v, ok := merged.ProviderOptions["image_size"].(string); ok && v != "" {
			size = v
		}
	}

	req := openai.ImageRequest{
		Model:          model,
		Prompt:         prompt,
		Size:           size,
		N:              1,
		ResponseFormat: openai.CreateImageResponseFormatB64JSON,
	}

	resp, err := o.client.CreateImage(ctx, req)
	if err != nil {
		if o.logger != nil {
			o.logger.Error("OpenAI image generation error",
				slog.String("error", err.Error()),
				slog.String("model", model))
		} else if o.verbose {
			fmt.Printf("OpenAI image generation error: %v\n", err)
		}
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no image generated")
	}

	imageData := strings.TrimSpace(resp.Data[0].B64JSON)
	if imageData == "" {
		return nil, fmt.Errorf("image payload missing in response")
	}

	bytes, err := base64.StdEncoding.DecodeString(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image data: %w", err)
	}

	return bytes, nil
}

// GenerateEmbedding implements LlmInterface
func (o *openaiImplementation) GenerateEmbedding(text string) ([]float32, error) {
	ctx := context.Background()

	// Use the configured model if set, otherwise fall back to Ada
	embeddingModel := openai.EmbeddingModel(o.model)
	if o.model == "" {
		embeddingModel = openai.AdaEmbeddingV2
	}

	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: embeddingModel,
	}

	resp, err := o.client.CreateEmbeddings(ctx, req)
	if err != nil {
		if o.logger != nil {
			o.logger.Error("OpenAI embedding generation error",
				slog.String("error", err.Error()))
		} else if o.verbose {
			fmt.Printf("OpenAI embedding generation error: %v\n", err)
		}
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings generated")
	}

	return resp.Data[0].Embedding, nil
}
