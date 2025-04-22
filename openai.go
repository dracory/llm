package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OpenAI model constants
const (
	OpenAIModelGPT35Turbo = "gpt-3.5-turbo"
	OpenAIModelGPT4       = "gpt-4"
	OpenAIModelGPT4Turbo  = "gpt-4-turbo-preview"
	OpenAIModelGPT4Vision = "gpt-4-vision-preview"
	OpenAIModelGPT4OMini  = "gpt-4o-mini"
	OpenAIModelGPT4O      = "gpt-4o"
)

// openaiImplementation implements ModelInterface for OpenAI
type openaiImplementation struct {
	client  *openai.Client
	options ModelOptions
}

// newOpenAIModel creates a new OpenAI provider implementation
func newOpenAIModel(options ModelOptions) (ModelInterface, error) {
	if options.ApiKey == "" {
		return nil, fmt.Errorf("openai API key not provided")
	}

	// Create OpenAI client
	client := openai.NewClient(options.ApiKey)

	// Default to GPT-4 Turbo if model is not specified
	if options.Model == "" {
		options.Model = OpenAIModelGPT4Turbo
	}

	return &openaiImplementation{
		client:  client,
		options: options,
	}, nil
}

// Complete implements the ModelInterface
func (o *openaiImplementation) Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error) {
	// Configure response format based on output format
	responseFormat := &openai.ChatCompletionResponseFormat{}
	switch o.options.OutputFormat {
	case OutputFormatJSON:
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeJSONObject
	default:
		responseFormat.Type = openai.ChatCompletionResponseFormatTypeText
	}

	// Set up messages with system and user prompts
	messages := []openai.ChatCompletionMessage{}

	// Add system message if provided
	if request.SystemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: request.SystemPrompt,
		})
	}

	// Add user message
	if request.UserPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: request.UserPrompt,
		})
	}

	// Default max tokens if not specified in request
	maxTokens := o.options.MaxTokens
	if request.MaxTokens > 0 {
		maxTokens = request.MaxTokens
	}

	// Default temperature if not specified in request
	temperature := o.options.Temperature
	if request.Temperature > 0 {
		temperature = request.Temperature
	}

	// Create request
	req := openai.ChatCompletionRequest{
		Model:          o.options.Model,
		ResponseFormat: responseFormat,
		Messages:       messages,
		MaxTokens:      maxTokens,
		Temperature:    float32(temperature),
	}

	// Generate response
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if o.options.Verbose {
			fmt.Printf("OpenAI generation error: %v\n", err)
		}
		return CompletionResponse{}, err
	}

	if len(resp.Choices) == 0 {
		return CompletionResponse{}, fmt.Errorf("no response from OpenAI")
	}

	// Extract the response text and trim whitespace
	result := strings.TrimSpace(resp.Choices[0].Message.Content)

	// Get tokens used from response
	tokensUsed := resp.Usage.TotalTokens

	return CompletionResponse{
		Text:       result,
		TokensUsed: tokensUsed,
	}, nil
}

// GetProvider implements ModelInterface
func (o *openaiImplementation) GetProvider() Provider {
	return o.options.Provider
}

// GetOutputFormat implements ModelInterface
func (o *openaiImplementation) GetOutputFormat() OutputFormat {
	return o.options.OutputFormat
}

// GetApiKey implements ModelInterface
func (o *openaiImplementation) GetApiKey() string {
	return o.options.ApiKey
}

// GetModel implements ModelInterface
func (o *openaiImplementation) GetModel() string {
	return o.options.Model
}

// GetMaxTokens implements ModelInterface
func (o *openaiImplementation) GetMaxTokens() int {
	return o.options.MaxTokens
}

// GetTemperature implements ModelInterface
func (o *openaiImplementation) GetTemperature() float64 {
	return o.options.Temperature
}

// GetProjectID implements ModelInterface
func (o *openaiImplementation) GetProjectID() string {
	return o.options.ProjectID
}

// GetRegion implements ModelInterface
func (o *openaiImplementation) GetRegion() string {
	return o.options.Region
}

// GetVerbose implements ModelInterface
func (o *openaiImplementation) GetVerbose() bool {
	return o.options.Verbose
}
