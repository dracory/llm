package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/samber/lo"
)

type customImplementation struct {
	apiKey      string
	endpointURL string
	model       string
	maxTokens   int
	temperature float64
	verbose     bool
	httpClient  *http.Client
}

func newCustomImplementation(options LlmOptions) (LlmInterface, error) {
	apiKey := strings.TrimSpace(options.ApiKey)

	endpointURL := ""
	if options.ProviderOptions != nil {
		if v, ok := options.ProviderOptions["url"].(string); ok {
			endpointURL = strings.TrimSpace(v)
		}
		if endpointURL == "" {
			if v, ok := options.ProviderOptions["endpoint_url"].(string); ok {
				endpointURL = strings.TrimSpace(v)
			}
		}
		if endpointURL == "" {
			if v, ok := options.ProviderOptions["base_url"].(string); ok {
				endpointURL = strings.TrimSpace(v)
			}
		}
	}

	if endpointURL == "" {
		return nil, fmt.Errorf("endpoint url is required")
	}

	model := strings.TrimSpace(options.Model)
	if model == "" {
		model = "default"
	}

	client := &http.Client{}

	return &customImplementation{
		apiKey:      apiKey,
		endpointURL: endpointURL,
		model:       model,
		maxTokens:   options.MaxTokens,
		temperature: options.Temperature,
		verbose:     options.Verbose,
		httpClient:  client,
	}, nil
}

func (c *customImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	merged := mergeOptions(LlmOptions{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Temperature: c.temperature,
		Verbose:     c.verbose,
		ProviderOptions: map[string]any{
			"url": c.endpointURL,
		},
	}, options)

	endpointURL := c.endpointURL
	if merged.ProviderOptions != nil {
		if v, ok := merged.ProviderOptions["url"].(string); ok {
			if s := strings.TrimSpace(v); s != "" {
				endpointURL = s
			}
		}
		if endpointURL == "" {
			if v, ok := merged.ProviderOptions["endpoint_url"].(string); ok {
				endpointURL = strings.TrimSpace(v)
			}
		}
		if endpointURL == "" {
			if v, ok := merged.ProviderOptions["base_url"].(string); ok {
				endpointURL = strings.TrimSpace(v)
			}
		}
	}
	if endpointURL == "" {
		return "", fmt.Errorf("endpoint url is required")
	}

	model := c.model
	if merged.Model != "" {
		model = merged.Model
	}

	maxTokens := c.maxTokens
	if merged.MaxTokens > 0 {
		maxTokens = merged.MaxTokens
	}

	temperature := c.temperature
	if merged.Temperature > 0 {
		temperature = merged.Temperature
	}

	responseFormat := "text"
	if merged.OutputFormat == OutputFormatJSON {
		responseFormat = "json_object"
	}

	type requestMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type requestBody struct {
		Model          string           `json:"model"`
		Messages       []requestMessage `json:"messages"`
		MaxTokens      int              `json:"max_tokens,omitempty"`
		Temperature    float64          `json:"temperature,omitempty"`
		ResponseFormat map[string]any   `json:"response_format,omitempty"`
	}

	body := requestBody{
		Model: model,
		Messages: []requestMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userMessage},
		},
		MaxTokens:   maxTokens,
		Temperature: temperature,
		ResponseFormat: map[string]any{
			"type": responseFormat,
		},
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	ctx := context.Background()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpointURL, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	if strings.TrimSpace(c.apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("request to %s failed: %w", endpointURL, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf(
			"request to %s failed with status %d: %s",
			endpointURL,
			resp.StatusCode,
			string(respBody),
		)
	}

	// OpenAI-compatible response
	type responseMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type responseChoice struct {
		Message responseMessage `json:"message"`
	}
	type responseRoot struct {
		Choices []responseChoice `json:"choices"`
	}

	var parsed responseRoot
	if err := json.Unmarshal(respBody, &parsed); err == nil {
		if len(parsed.Choices) > 0 {
			return strings.TrimSpace(parsed.Choices[0].Message.Content), nil
		}
	}

	// Fallback: allow plain-text responses
	return strings.TrimSpace(string(respBody)), nil
}

func (c *customImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *customImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *customImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	return nil, fmt.Errorf("image generation not supported by custom provider")
}

func (c *customImplementation) GenerateEmbedding(text string) ([]float32, error) {
	return nil, fmt.Errorf("embedding generation not supported by custom provider")
}

// Optional helper for providers that return base64-encoded images in their content.
func decodeBase64Image(data string) ([]byte, error) {
	if strings.TrimSpace(data) == "" {
		return nil, fmt.Errorf("empty image data")
	}
	return base64.StdEncoding.DecodeString(data)
}
