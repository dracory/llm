package llm

import (
	"context"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"github.com/samber/lo"
	"google.golang.org/api/option"
)

// Example usage:
//
// llmEngine, err := llm.NewGemini(llm.LlmOptions{
// 	ApiKey:    config.GoogleGeminiApiKey,
// 	MaxTokens: 4096, // Suitable for blog post generation
// 	Verbose:   config.Debug,
// })
// if err != nil {
// 	return ctx, data, errors.New("failed to initialize LLM engine")
// }

// geminiImplementation implements LlmInterface for Gemini
type geminiImplementation struct {
	client    *genai.Client
	model     *genai.GenerativeModel
	maxTokens int
	verbose   bool
}

// newGeminiImplementation creates a new Gemini provider implementation
func newGeminiImplementation(options LlmOptions) (LlmInterface, error) {
	if options.ApiKey == "" {
		return nil, fmt.Errorf("Gemini API key not provided")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(options.ApiKey))
	if err != nil {
		if options.Verbose {
			fmt.Printf("Failed to create Gemini client: %v\n", err)
		}
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	// Default to Gemini Flash model
	var model *genai.GenerativeModel
	if options.Model != "" {
		model = client.GenerativeModel(options.Model)
	} else {
		model = client.GenerativeModel(GEMINI_MODEL_2_5_FLASH)
	}

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
		client:    client,
		model:     model,
		maxTokens: options.MaxTokens,
		verbose:   options.Verbose,
	}, nil
}

// Generate implements LlmInterface
func (g *geminiImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	if g.client == nil {
		return "", fmt.Errorf("Gemini client not initialized")
	}

	ctx := context.Background()

	// Combine system prompt and user message
	prompt := fmt.Sprintf("%s\n\n%s", systemPrompt, userMessage)
	generationConfig := genai.GenerationConfig{}

	if options.MaxTokens > 0 || options.Temperature > 0 {
		generationConfig.MaxOutputTokens = int32Ptr(options.MaxTokens)
		generationConfig.Temperature = float32Ptr(float32(options.Temperature))
	}

	if options.OutputFormat == OutputFormatText {
		generationConfig.ResponseMIMEType = "text/plain"
	}

	if options.OutputFormat == OutputFormatJSON {
		generationConfig.ResponseMIMEType = "application/json"
	}

	if options.OutputFormat == OutputFormatXML {
		generationConfig.ResponseMIMEType = "application/xml"
	}

	if options.OutputFormat == OutputFormatYAML {
		generationConfig.ResponseMIMEType = "application/yaml"
	}

	if options.OutputFormat == OutputFormatEnum {
		generationConfig.ResponseMIMEType = "text/x.enum"
	}

	if options.OutputFormat == OutputFormatImagePNG || options.OutputFormat == OutputFormatImageJPG {
		generationConfig.ResponseMIMEType = "image/png"
	}

	g.model.GenerationConfig = generationConfig

	// Generate response
	resp, err := g.model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		if g.verbose {
			fmt.Printf("Gemini generation error: %v\n", err)
		}
		return "", err
	}

	if len(resp.Candidates) == 0 {
		return "", fmt.Errorf("no response from Gemini")
	}

	// Get the text from the first candidate
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			result += string(text)
		}
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
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	if options.OutputFormat != OutputFormatImagePNG && options.OutputFormat != OutputFormatImageJPG {
		options.OutputFormat = OutputFormatImagePNG
	}

	raw, err := g.Generate(prompt, "", options)

	if err != nil {
		return nil, err
	}

	return []byte(raw), nil
}

func int32Ptr(i int) *int32 {
	i32 := int32(i)
	return &i32
}
func float32Ptr(f float32) *float32 {
	return &f
}
