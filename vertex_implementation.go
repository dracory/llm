package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"github.com/mingrammer/cfmt"
	"github.com/samber/lo"
	"github.com/spf13/cast"
	"google.golang.org/api/option"
)

// const GEMINI_MODEL_2_0_FLASH = "gemini-2.0-flash-001"
// const GEMINI_MODEL_2_0_FLASH_LITE = "gemini-2.0-flash-lite-001"
const GEMINI_MODEL_2_0_FLASH_EXP_IMAGE_GENERATION = "gemini-2.0-flash-exp-image-generation"

const GEMINI_MODEL_2_5_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
const GEMINI_MODEL_2_5_FLASH = "gemini-2.5-flash"
const GEMINI_MODEL_2_5_PRO = "gemini-2.5-pro"

const GEMINI_MODEL_1_5_PRO = "gemini-1.5-pro"             // supported but now old
const GEMINI_MODEL_1_5_FLASH = "gemini-1.5-flash"         // supported but now old
const GEMINI_MODEL_3_0_IMAGEN = "imagen-3.0-generate-002" // not supported

func newVertexImplementation(options LlmOptions) (LlmInterface, error) {
	o := options
	// Add checks for required options if needed, e.g. API key
	return &vertexLlmImpl{
		options: o,
	}, nil
}

type vertexLlmImpl struct {
	options LlmOptions
}

// Generate generates a response from the LLM based on the provided system prompt and user message.
// It merges the provided options with the default options and returns the generated response.
// This allows the user to override the default options.
func (c *vertexLlmImpl) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options = mergeOptions(c.options, options)

	if options.ProjectID == "" {
		return "", errors.New("project id is required")
	}

	if options.Region == "" {
		return "", errors.New("region is required")
	}

	ctx := context.Background()
	vertexCredentialsJSON := `vertexapicredentials.json` // best move to vault
	vertxCredentialsContent, err := os.ReadFile(vertexCredentialsJSON)
	if err != nil {
		return "", err
	}
	client, err := genai.NewClient(ctx, options.ProjectID, options.Region, option.WithCredentialsJSON([]byte(vertxCredentialsContent)))
	if err != nil {
		return "", err
	}
	defer client.Close()

	systemMessage := "Hi. I'll explain how you should behave:\n" + systemPrompt

	var final string
	if options.OutputFormat == OutputFormatJSON {
		final = systemMessage + "\n\nUSER:" + userMessage + "\n\nYou must respond with a JSON object only. Do not include any text outside the JSON."
	} else {
		final = systemMessage + "\n\nUSER:" + userMessage + "\n\nDo not use markdown."
	}

	if options.Verbose {
		cfmt.Warningln("Final prompt:", final)
	}

	// For text-only input, use the gemini-pro model
	model := client.GenerativeModel(findVertexModelName(options.Model))

	// Convert values to pointers for generation config
	temp := float32(options.Temperature)
	maxTokens := int32(options.MaxTokens)
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

	if options.OutputFormat == OutputFormatJSON {
		generationConfig.ResponseMIMEType = "application/json"
	} else if options.OutputFormat == OutputFormatXML {
		generationConfig.ResponseMIMEType = "application/xml"
	} else if options.OutputFormat == OutputFormatYAML {
		generationConfig.ResponseMIMEType = "application/yaml"
	} else if options.OutputFormat == OutputFormatEnum {
		generationConfig.ResponseMIMEType = "text/x.enum"
	} else {
		generationConfig.ResponseMIMEType = "text/plain"
	}
	model.GenerationConfig = *generationConfig

	// Configure safety settings for JSON output
	if options.OutputFormat == OutputFormatJSON {
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

	resp, err := model.GenerateContent(ctx, genai.Text(final))
	if err != nil {
		return "", err
	}

	// Parse response
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) != 1 {
		return "", err
	}

	str := cast.ToString(resp.Candidates[0].Content.Parts[0])
	return strings.TrimSpace(str), nil
}

func (l *vertexLlmImpl) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options = mergeOptions(l.options, options)
	options.OutputFormat = OutputFormatText
	return l.Generate(systemPrompt, userPrompt, options)
}

func (l *vertexLlmImpl) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options = mergeOptions(l.options, options)
	options.OutputFormat = OutputFormatJSON
	return l.Generate(systemPrompt, userPrompt, options)
}

func (l *vertexLlmImpl) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options = mergeOptions(l.options, options)
	ctx := context.Background()
	client, err := genai.NewClient(ctx, l.options.ProjectID, l.options.Region)
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}
	defer client.Close()

	cfmt.Warningln("Using experimental image generation model")
	cfmt.Warningln("Prompt:", prompt)

	model := client.GenerativeModel(GEMINI_MODEL_2_0_FLASH_EXP_IMAGE_GENERATION)

	// Convert values to pointers for generation config
	temp := float32(options.Temperature)
	maxTokens := int32(options.MaxTokens)
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
	if options.OutputFormat == OutputFormatImagePNG {
		generationConfig.ResponseMIMEType = "image/png"
	} else if options.OutputFormat == OutputFormatImageJPG {
		generationConfig.ResponseMIMEType = "image/jpg"
	} else {
		generationConfig.ResponseMIMEType = "image/png"
	}
	model.GenerationConfig = *generationConfig
	resp, err := model.GenerateContent(ctx,
		genai.Text(prompt),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("no image generated")
	}

	for _, part := range resp.Candidates[0].Content.Parts {
		if blob, ok := part.(genai.Blob); ok && blob.MIMEType == "image/png" {
			return blob.Data, nil
		}
	}

	return nil, errors.New("no image found in response")
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
//
// Parameters:
// - model: the name of the model
//
// Returns:
// - the name of the model
func findVertexModelName(modelName string) string {
	if strings.Contains(modelName, "pro") {
		//return GEMINI_MODEL_2_0_FLASH
		return GEMINI_MODEL_2_5_PRO
	}

	// return GEMINI_MODEL_2_0_FLASH_LITE
	return GEMINI_MODEL_2_5_FLASH
}
