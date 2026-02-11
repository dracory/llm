package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
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
	logger      *slog.Logger
	apiKey      string
	baseURL     string
	httpClient  openai.HTTPDoer
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

	baseURL := "https://openrouter.ai/api/v1"

	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL

	client := openai.NewClientWithConfig(cfg)

	return &openrouterImplementation{
		client:      client,
		model:       model,
		maxTokens:   o.MaxTokens,
		temperature: o.Temperature,
		verbose:     o.Verbose,
		logger:      o.Logger,
		apiKey:      apiKey,
		baseURL:     baseURL,
		httpClient:  cfg.HTTPClient,
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

	verbose := o.verbose
	if options.Verbose {
		verbose = options.Verbose
	}

	// Configure response format based on output format
	responseFormat := &openai.ChatCompletionResponseFormat{}
	if options.OutputFormat == OutputFormatJSON {
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeJSONObject
	} else {
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeText
	}

	if o.logger != nil {
		o.logger.Debug("OpenRouter request",
			slog.String("model", model),
			slog.Int("max_tokens", maxTokens),
			slog.Float64("temperature", temperature),
			slog.Int("system_prompt_len", len(systemPrompt)),
			slog.Int("user_message_len", len(userMessage)))
	} else if verbose {
		fmt.Printf("OpenRouter request: model=%s, maxTokens=%d, temperature=%f\n", model, maxTokens, temperature)
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
		if o.logger != nil {
			o.logger.Error("OpenRouter API request failed",
				slog.String("error", err.Error()),
				slog.String("model", model),
				slog.String("base_url", o.baseURL))
		} else if verbose {
			fmt.Printf("OpenRouter generation error: %v\n", err)
		}
		return "", err
	}

	if o.logger != nil {
		o.logger.Debug("OpenRouter response received",
			slog.String("model", model))
	} else if verbose {
		fmt.Printf("OpenRouter response received: model=%s\n", model)
	}

	if len(resp.Choices) == 0 {
		if o.logger != nil {
			o.logger.Warn("no response from OpenRouter",
				slog.String("model", model))
		} else if verbose {
			fmt.Printf("no response from OpenRouter: model=%s\n", model)
		}
		return "", fmt.Errorf("no response from OpenRouter")
	}

	response := resp.Choices[0].Message.Content
	if o.logger != nil {
		o.logger.Debug("OpenRouter response content",
			slog.String("model", model),
			slog.Int("length", len(response)))
	} else if verbose {
		fmt.Printf("OpenRouter response: length=%d\n", len(response))
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
// OpenRouter uses the chat completions endpoint with modalities parameter for image generation
func (o *openrouterImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	options := lo.FirstOr(opts, LlmOptions{})

	ctx := context.Background()

	// Apply options if provided
	model := o.model
	if options.Model != "" {
		model = options.Model
	}

	verbose := o.verbose
	if options.Verbose {
		verbose = options.Verbose
	}

	if o.logger != nil {
		o.logger.Debug("OpenRouter image generation request",
			slog.String("model", model),
			slog.Int("prompt_len", len(prompt)))
	} else if verbose {
		fmt.Printf("OpenRouter image generation request: model=%s\n", model)
	}

	// OpenRouter requires using chat completions with modalities for image generation
	// We need to use a custom request structure that includes modalities
	type imageConfig struct {
		AspectRatio string `json:"aspect_ratio,omitempty"`
	}

	type chatRequest struct {
		Model       string                         `json:"model"`
		Messages    []openai.ChatCompletionMessage `json:"messages"`
		Modalities  []string                       `json:"modalities"`
		ImageConfig *imageConfig                   `json:"image_config,omitempty"`
	}

	// Create the request with modalities
	reqBody := chatRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		Modalities: []string{"image", "text"},
		ImageConfig: &imageConfig{
			AspectRatio: "1:1", // Default to square images
		},
	}

	// We need to make a custom HTTP request since the standard client doesn't support modalities
	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/chat/completions", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 100<<20))
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("image generation failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse the response to extract the image
	type imageURL struct {
		URL string `json:"url"`
	}

	type imageData struct {
		Type     string   `json:"type"`
		ImageURL imageURL `json:"image_url"`
	}

	type message struct {
		Role    string      `json:"role"`
		Content string      `json:"content"`
		Images  []imageData `json:"images"`
	}

	type choice struct {
		Message message `json:"message"`
	}

	type chatResponse struct {
		Choices []choice `json:"choices"`
	}

	var chatResp chatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	if len(chatResp.Choices[0].Message.Images) == 0 {
		return nil, fmt.Errorf("no images in response")
	}

	// Extract the base64 image data from the data URL
	dataURL := chatResp.Choices[0].Message.Images[0].ImageURL.URL
	if !strings.HasPrefix(dataURL, "data:image/") {
		return nil, fmt.Errorf("unexpected image URL format: %s", dataURL)
	}

	// Extract base64 data from data URL (format: data:image/png;base64,...)
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid data URL format")
	}

	imageBytes, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 image: %w", err)
	}

	if o.logger != nil {
		o.logger.Debug("Successfully generated image",
			slog.Int("bytes", len(imageBytes)))
	} else if verbose {
		fmt.Printf("Successfully generated image: %d bytes\n", len(imageBytes))
	}

	return imageBytes, nil
}

func (o *openrouterImplementation) GenerateEmbedding(text string) ([]float32, error) {
	ctx := context.Background()

	// OpenRouter uses OpenAI-compatible embeddings endpoint
	// Use the configured model if set, otherwise fall back to Ada
	embeddingModel := openai.EmbeddingModel(o.model)
	if o.model == "" || o.model == "openrouter/auto" {
		embeddingModel = openai.AdaEmbeddingV2
	}

	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: embeddingModel,
	}

	resp, err := o.client.CreateEmbeddings(ctx, req)
	if err != nil {
		if o.logger != nil {
			o.logger.Error("OpenRouter embedding generation error",
				slog.String("error", err.Error()))
		} else if o.verbose {
			fmt.Printf("OpenRouter embedding generation error: %v\n", err)
		}
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings generated")
	}

	return resp.Data[0].Embedding, nil
}
