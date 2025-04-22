package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"google.golang.org/api/option"
)

// Vertex AI model constants
const (
	VertexModelGemini20Flash         = "gemini-2.0-flash-001"
	VertexModelGemini20FlashLite     = "gemini-2.0-flash-lite-001"
	VertexModelGemini20FlashImageGen = "gemini-2.0-flash-exp-image-generation"
	VertexModelGemini25ProPreview    = "gemini-2.5-pro-preview-03-25"
	VertexModelGemini15Pro           = "gemini-1.5-pro"   // supported but older
	VertexModelGemini15Flash         = "gemini-1.5-flash" // supported but older
)

// vertexImplementation implements ModelInterface for Vertex AI
type vertexImplementation struct {
	options ModelOptions
}

// newVertexModel creates a new Vertex AI provider implementation
func newVertexModel(options ModelOptions) (ModelInterface, error) {
	// Validate required options
	if options.ProjectID == "" {
		return nil, fmt.Errorf("vertex AI project ID is required")
	}

	if options.Region == "" {
		options.Region = "us-central1" // Default region if not specified
	}

	// Default to Gemini 2.0 Flash Lite if model is not specified
	if options.Model == "" {
		options.Model = VertexModelGemini20FlashLite
	}

	return &vertexImplementation{
		options: options,
	}, nil
}

// Complete implements the ModelInterface
func (v *vertexImplementation) Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error) {
	// Check for required fields
	if v.options.ProjectID == "" {
		return CompletionResponse{}, errors.New("project ID is required")
	}

	if v.options.Region == "" {
		return CompletionResponse{}, errors.New("region is required")
	}

	// Initialize client options
	var clientOptions []option.ClientOption

	// If API key or credentials JSON is provided, use it
	if v.options.ApiKey != "" {
		clientOptions = append(clientOptions, option.WithAPIKey(v.options.ApiKey))
	}

	// Create Vertex AI client
	client, err := genai.NewClient(ctx, v.options.ProjectID, v.options.Region, clientOptions...)
	if err != nil {
		if v.options.Verbose {
			fmt.Printf("Failed to create Vertex AI client: %v\n", err)
		}
		return CompletionResponse{}, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}
	defer client.Close()

	// Prepare prompts
	systemPrompt := ""
	if request.SystemPrompt != "" {
		systemPrompt = "Hi. I'll explain how you should behave:\n" + request.SystemPrompt
	}

	userPrompt := request.UserPrompt

	var finalPrompt string
	if systemPrompt != "" && userPrompt != "" {
		if v.options.OutputFormat == OutputFormatJSON {
			finalPrompt = systemPrompt + "\n\nUSER:" + userPrompt + "\n\nYou must respond with a JSON object only. Do not include any text outside the JSON."
		} else {
			finalPrompt = systemPrompt + "\n\nUSER:" + userPrompt
		}
	} else if systemPrompt != "" {
		finalPrompt = systemPrompt
	} else {
		finalPrompt = userPrompt
	}

	if v.options.Verbose {
		fmt.Printf("Vertex AI prompt: %s\n", finalPrompt)
	}

	// Get the appropriate model
	modelName := findVertexModelName(v.options.Model)
	model := client.GenerativeModel(modelName)

	// Set up generation config
	maxTokens := int32(request.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = int32(v.options.MaxTokens)
	}

	temp := float32(request.Temperature)
	if temp <= 0 {
		temp = float32(v.options.Temperature)
	}

	candidateCount := int32(1)
	topP := float32(0.8)
	topK := int32(40)

	// Configure generation parameters
	generationConfig := &genai.GenerationConfig{
		Temperature:     &temp,
		MaxOutputTokens: &maxTokens,
		CandidateCount:  &candidateCount,
		TopP:            &topP,
		TopK:            &topK,
	}

	// Set response format based on output format
	switch v.options.OutputFormat {
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

	model.GenerationConfig = *generationConfig

	// Configure safety settings for JSON output
	if v.options.OutputFormat == OutputFormatJSON {
		safetySettings := []*genai.SafetySetting{
			{
				Category:  genai.HarmCategoryHarassment,
				Threshold: genai.HarmBlockLowAndAbove,
			},
			{
				Category:  genai.HarmCategoryHateSpeech,
				Threshold: genai.HarmBlockLowAndAbove,
			},
			{
				Category:  genai.HarmCategoryDangerousContent,
				Threshold: genai.HarmBlockLowAndAbove,
			},
			{
				Category:  genai.HarmCategorySexuallyExplicit,
				Threshold: genai.HarmBlockLowAndAbove,
			},
		}
		model.SafetySettings = safetySettings
	}

	// Generate response
	resp, err := model.GenerateContent(ctx, genai.Text(finalPrompt))
	if err != nil {
		if v.options.Verbose {
			fmt.Printf("Vertex AI generation error: %v\n", err)
		}
		return CompletionResponse{}, err
	}

	// Parse response
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return CompletionResponse{}, errors.New("no response from Vertex AI")
	}

	// Extract text from response
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			result += string(text)
		}
	}

	result = strings.TrimSpace(result)

	// Approximate token count
	tokensUsed := CountTokens(result)

	return CompletionResponse{
		Text:       result,
		TokensUsed: tokensUsed,
	}, nil
}

// GetProvider implements ModelInterface
func (v *vertexImplementation) GetProvider() Provider {
	return v.options.Provider
}

// GetOutputFormat implements ModelInterface
func (v *vertexImplementation) GetOutputFormat() OutputFormat {
	return v.options.OutputFormat
}

// GetApiKey implements ModelInterface
func (v *vertexImplementation) GetApiKey() string {
	return v.options.ApiKey
}

// GetModel implements ModelInterface
func (v *vertexImplementation) GetModel() string {
	return v.options.Model
}

// GetMaxTokens implements ModelInterface
func (v *vertexImplementation) GetMaxTokens() int {
	return v.options.MaxTokens
}

// GetTemperature implements ModelInterface
func (v *vertexImplementation) GetTemperature() float64 {
	return v.options.Temperature
}

// GetProjectID implements ModelInterface
func (v *vertexImplementation) GetProjectID() string {
	return v.options.ProjectID
}

// GetRegion implements ModelInterface
func (v *vertexImplementation) GetRegion() string {
	return v.options.Region
}

// GetVerbose implements ModelInterface
func (v *vertexImplementation) GetVerbose() bool {
	return v.options.Verbose
}

// findVertexModelName returns the name of the gemini model to use
// based on the model name.
//
// Supported models by gemini are:
// - gemini-1.5-flash is the default model
// - gemini-1.5-pro is the pro model, it is used if the model name contains "pro"
// - gemini-2.0-flash is the flash model, it is used if the model name contains "flash"
// - gemini-2.0-flash-lite is the flash-lite model, it is used if the model name contains "flash-lite"
// - gemini-2.5-pro-preview-03-25 is the pro preview model, it is used if the model name contains "pro-preview"
//
// See https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning
// for details on model naming and versioning
func findVertexModelName(modelName string) string {
	if modelName == "" {
		return VertexModelGemini20FlashLite
	}

	if strings.Contains(strings.ToLower(modelName), "pro-preview") {
		return VertexModelGemini25ProPreview
	}

	if strings.Contains(strings.ToLower(modelName), "pro") {
		return VertexModelGemini15Pro
	}

	if strings.Contains(strings.ToLower(modelName), "flash-lite") {
		return VertexModelGemini20FlashLite
	}

	if strings.Contains(strings.ToLower(modelName), "flash") {
		return VertexModelGemini20Flash
	}

	// If no match, return the model name as is
	return modelName
}
