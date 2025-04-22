package llm

import (
	"context"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Gemini model constants
const (
	GeminiModel1Pro   = "gemini-pro"
	GeminiModel1Flash = "gemini-pro-flash"
	GeminiModel2Pro   = "gemini-2-pro"
	GeminiModel2Flash = "gemini-2-flash"
)

// geminiImplementation implements ModelInterface for Gemini
type geminiImplementation struct {
	client  *genai.Client
	model   *genai.GenerativeModel
	options ModelOptions
}

// newGeminiModel creates a new Gemini provider implementation
func newGeminiModel(options ModelOptions) (ModelInterface, error) {
	if options.ApiKey == "" {
		return nil, fmt.Errorf("gemini API key not provided")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(options.ApiKey))
	if err != nil {
		if options.Verbose {
			fmt.Printf("Failed to create Gemini client: %v\n", err)
		}
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	// Default to Gemini 2 Flash model if not specified
	modelName := options.Model
	if modelName == "" {
		modelName = GeminiModel2Flash
	}

	model := client.GenerativeModel(modelName)

	// Set safety settings to default (allow most content)
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockNone,
		},
	}

	return &geminiImplementation{
		client:  client,
		model:   model,
		options: options,
	}, nil
}

// Complete implements the ModelInterface
func (g *geminiImplementation) Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error) {
	if g.client == nil {
		return CompletionResponse{}, fmt.Errorf("gemini client not initialized")
	}

	generationConfig := genai.GenerationConfig{}

	// Apply max tokens and temperature if specified
	if g.options.MaxTokens > 0 {
		maxTokens := int32(g.options.MaxTokens)
		generationConfig.MaxOutputTokens = &maxTokens
	}

	if g.options.Temperature > 0 {
		temp := float32(g.options.Temperature)
		generationConfig.Temperature = &temp
	}

	// Set response MIME type based on output format
	switch g.options.OutputFormat {
	case OutputFormatJSON:
		generationConfig.ResponseMIMEType = "application/json"
	case OutputFormatXML:
		generationConfig.ResponseMIMEType = "application/xml"
	case OutputFormatYAML:
		generationConfig.ResponseMIMEType = "application/yaml"
	case OutputFormatEnum:
		generationConfig.ResponseMIMEType = "text/x.enum"
	case OutputFormatImagePNG:
		generationConfig.ResponseMIMEType = "image/png"
	case OutputFormatImageJPG:
		generationConfig.ResponseMIMEType = "image/jpeg"
	default:
		generationConfig.ResponseMIMEType = "text/plain"
	}

	g.model.GenerationConfig = generationConfig

	// Prepare the prompt by combining system and user prompts
	var prompt string
	if request.SystemPrompt != "" && request.UserPrompt != "" {
		// Both present - format as system instructions followed by user query
		prompt = fmt.Sprintf("%s\n\n%s", request.SystemPrompt, request.UserPrompt)
	} else if request.SystemPrompt != "" {
		// Only system prompt provided
		prompt = request.SystemPrompt
	} else {
		// Only user prompt or both are empty
		prompt = request.UserPrompt
	}

	// Generate response
	resp, err := g.model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		if g.options.Verbose {
			fmt.Printf("Gemini generation error: %v\n", err)
		}
		return CompletionResponse{}, err
	}

	if len(resp.Candidates) == 0 {
		return CompletionResponse{}, fmt.Errorf("no response from Gemini")
	}

	// Get the text from the first candidate
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			result += string(text)
		}
	}

	// Estimate tokens used - this is approximate since Gemini doesn't always return token count
	tokensUsed := CountTokens(result)

	return CompletionResponse{
		Text:       result,
		TokensUsed: tokensUsed,
	}, nil
}

// GetProvider implements ModelInterface
func (g *geminiImplementation) GetProvider() Provider {
	return g.options.Provider
}

// GetOutputFormat implements ModelInterface
func (g *geminiImplementation) GetOutputFormat() OutputFormat {
	return g.options.OutputFormat
}

// GetApiKey implements ModelInterface
func (g *geminiImplementation) GetApiKey() string {
	return g.options.ApiKey
}

// GetModel implements ModelInterface
func (g *geminiImplementation) GetModel() string {
	return g.options.Model
}

// GetMaxTokens implements ModelInterface
func (g *geminiImplementation) GetMaxTokens() int {
	return g.options.MaxTokens
}

// GetTemperature implements ModelInterface
func (g *geminiImplementation) GetTemperature() float64 {
	return g.options.Temperature
}

// GetProjectID implements ModelInterface
func (g *geminiImplementation) GetProjectID() string {
	return g.options.ProjectID
}

// GetRegion implements ModelInterface
func (g *geminiImplementation) GetRegion() string {
	return g.options.Region
}

// GetVerbose implements ModelInterface
func (g *geminiImplementation) GetVerbose() bool {
	return g.options.Verbose
}
