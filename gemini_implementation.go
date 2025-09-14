package llm

import (
	"context"
	"fmt"

	"github.com/samber/lo"
	"google.golang.org/genai"
)

// geminiImplementation implements LlmInterface for Gemini
type geminiImplementation struct {
	client  *genai.Client
	model   string
	verbose bool
}

// newGeminiImplementation creates a new Gemini provider implementation
func newGeminiImplementation(options LlmOptions) (LlmInterface, error) {
	if options.ApiKey == "" {
		return nil, fmt.Errorf("Gemini API key not provided")
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
		client:  client,
		model:   modelName,
		verbose: options.Verbose,
	}, nil
}

// Generate implements LlmInterface
func (g *geminiImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	if g.client == nil {
		return "", fmt.Errorf("Gemini client not initialized")
	}

	// Prepare the prompt with system and user message
	prompt := systemPrompt
	if userMessage != "" {
		prompt += "\n\n" + userMessage
	}

	// Add format instructions if needed
	if options.OutputFormat == OutputFormatJSON {
		prompt += "\nYou must respond with valid JSON only. Do not include any text outside the JSON."
	}

	// Create a text part with the prompt
	textPart := &genai.Part{
		Text: prompt,
	}

	// Create content with the text part
	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{textPart},
	}

	// Prepare generation config if needed
	var genConfig *genai.GenerateContentConfig
	if options.MaxTokens > 0 || options.Temperature > 0 {
		genConfig = &genai.GenerateContentConfig{
			MaxOutputTokens: int32(options.MaxTokens),
		}
		if options.Temperature > 0 {
			genConfig.Temperature = genai.Ptr[float32](float32(options.Temperature))
		}
	}

	// Generate response
	resp, err := g.client.Models.GenerateContent(
		context.Background(),
		g.model,
		[]*genai.Content{content},
		genConfig,
	)

	if err != nil {
		if g.verbose {
			fmt.Printf("Gemini generation error: %v\n", err)
		}
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response from Gemini")
	}

	// Get the text from the first candidate
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			result += part.Text
		}
	}

	if result == "" {
		return "", fmt.Errorf("empty response from Gemini")
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
	// Add a specific instruction for JSON output
	systemPrompt += "\nYou must respond with valid JSON only. Do not include any text outside the JSON."
	return g.Generate(systemPrompt, userPrompt, options)
}

// GenerateImage implements LlmInterface
func (g *geminiImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	// Image generation is not directly supported in the current version of the Gemini API
	// You would need to use a different API like DALL-E or Stable Diffusion for image generation
	return nil, fmt.Errorf("image generation is not supported in this implementation")
}

func int32Ptr(i int) *int32 {
	i32 := int32(i)
	return &i32
}
func float32Ptr(f float32) *float32 {
	return &f
}
