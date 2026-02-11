package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/samber/lo"
	"google.golang.org/genai"
)

// geminiImplementation implements LlmInterface for Gemini
type geminiImplementation struct {
	client     *genai.Client
	model      string
	verbose    bool
	apiKey     string
	httpClient *http.Client
}

// newGeminiImplementation creates a new Gemini provider implementation
func newGeminiImplementation(options LlmOptions) (LlmInterface, error) {
	if options.ApiKey == "" {
		return nil, fmt.Errorf("gemini API key not provided")
	}

	// Create a new client with the API key
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  options.ApiKey,
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		if options.Verbose {
			fmt.Printf("Failed to create Gemini client: %v\n", err)
		}
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	// Default to Gemini Flash model
	modelName := GEMINI_MODEL_2_5_FLASH
	if options.Model != "" {
		modelName = options.Model
	}

	return &geminiImplementation{
		client:     client,
		model:      modelName,
		verbose:    options.Verbose,
		apiKey:     options.ApiKey,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// Generate implements LlmInterface
func (g *geminiImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	if g.client == nil {
		return "", fmt.Errorf("gemini client not initialized")
	}

	// Prepare user message content
	userContent := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: userMessage}},
	}

	// Prepare system instruction
	effectiveSystemPrompt := systemPrompt
	if options.OutputFormat == OutputFormatJSON {
		effectiveSystemPrompt += "\nYou must respond with valid JSON only. Do not include any text outside the JSON."
	}

	// Prepare generation config
	genConfig := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{{Text: effectiveSystemPrompt}},
		},
	}
	if options.MaxTokens > 0 {
		genConfig.MaxOutputTokens = int32(options.MaxTokens)
	}
	if options.Temperature > 0 {
		genConfig.Temperature = genai.Ptr(float32(options.Temperature))
	}

	// Generate response
	resp, err := g.client.Models.GenerateContent(
		context.Background(),
		g.model,
		[]*genai.Content{userContent},
		genConfig,
	)

	if err != nil {
		if g.verbose {
			fmt.Printf("Gemini generation error: %v\n", err)
		}
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response from gemini")
	}

	// Get the text from the first candidate
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			result += part.Text
		}
	}

	if result == "" {
		return "", fmt.Errorf("empty response from gemini")
	}

	return result, nil
}

// GenerateText implements LlmInterface
func (g *geminiImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return g.Generate(systemPrompt, userPrompt, options)
}

// GenerateJSON implements LlmInterface
func (g *geminiImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return g.Generate(systemPrompt, userPrompt, options)
}

// GenerateImage implements LlmInterface
func (g *geminiImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	// Image generation is not directly supported in the current version of the Gemini API
	// You would need to use a different API like DALL-E or Stable Diffusion for image generation
	return nil, fmt.Errorf("image generation is not supported in this implementation")
}

// GenerateEmbedding generates embeddings for the given text
func (g *geminiImplementation) GenerateEmbedding(text string) ([]float32, error) {
	ctx := context.Background()

	// Gemini requires a custom HTTP request for embeddings
	reqBody := map[string]interface{}{
		"model": "models/embedding-001",
		"text":  text,
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", g.apiKey)

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding struct {
			Value []float64 `json:"value"`
		} `json:"embedding"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(result.Embedding.Value) == 0 {
		return nil, fmt.Errorf("no embeddings generated")
	}

	// Convert float64 to float32
	embeddings := make([]float32, len(result.Embedding.Value))
	for i, v := range result.Embedding.Value {
		embeddings[i] = float32(v)
	}

	return embeddings, nil
}
