package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"github.com/samber/lo"
	"github.com/spf13/cast"
	"google.golang.org/api/option"
)

// const GEMINI_MODEL_2_0_FLASH = "gemini-2.0-flash-001"
// const GEMINI_MODEL_2_0_FLASH_LITE = "gemini-2.0-flash-lite-001"
const GEMINI_MODEL_2_0_FLASH_EXP_IMAGE_GENERATION = "gemini-2.0-flash-exp-image-generation"

const GEMINI_MODEL_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
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
	clientOptions, err := buildVertexClientOptions(options)
	if err != nil {
		return "", err
	}

	client, err := genai.NewClient(ctx, options.ProjectID, options.Region, clientOptions...)
	if err != nil {
		return "", err
	}
	defer func() {
		if cerr := client.Close(); cerr != nil {
			if options.Logger != nil {
				options.Logger.Warn("failed to close vertex client",
					slog.String("error", cerr.Error()))
			} else if options.Verbose {
				fmt.Printf("failed to close vertex client: %v\n", cerr)
			}
		}
	}()

	// Prepare system instruction
	effectiveSystemPrompt := systemPrompt
	if options.OutputFormat == OutputFormatJSON {
		effectiveSystemPrompt += "\nYou must respond with a JSON object only. Do not include any text outside the JSON."
	}

	if options.Logger != nil {
		options.Logger.Debug("Vertex AI request",
			slog.String("model", options.Model),
			slog.Int("system_prompt_len", len(effectiveSystemPrompt)),
			slog.Int("user_message_len", len(userMessage)))
	} else if options.Verbose {
		fmt.Printf("Vertex AI system prompt: %s\n", effectiveSystemPrompt)
		fmt.Printf("Vertex AI user message: %s\n", userMessage)
	}

	// For text-only input, use the gemini-pro model
	model := client.GenerativeModel(findVertexModelName(options.Model))

	// Set system instruction separately from user content
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{genai.Text(effectiveSystemPrompt)},
	}

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

	switch options.OutputFormat {
	case OutputFormatJSON:
		generationConfig.ResponseMIMEType = "application/json"
	case OutputFormatXML:
		generationConfig.ResponseMIMEType = "application/xml"
	case OutputFormatYAML:
		generationConfig.ResponseMIMEType = "application/yaml"
	case OutputFormatEnum:
		generationConfig.ResponseMIMEType = "text/x.enum"
	default:
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

	resp, err := model.GenerateContent(ctx, genai.Text(userMessage))
	if err != nil {
		return "", err
	}

	// Parse response
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("unexpected vertex response: no candidates or empty parts")
	}

	// Iterate over all parts and concatenate text parts
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		result += cast.ToString(part)
	}

	return strings.TrimSpace(result), nil
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

	if options.ProjectID == "" {
		return nil, errors.New("project id is required")
	}

	if options.Region == "" {
		return nil, errors.New("region is required")
	}

	ctx := context.Background()
	clientOptions, err := buildVertexClientOptions(options)
	if err != nil {
		return nil, err
	}

	client, err := genai.NewClient(ctx, options.ProjectID, options.Region, clientOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}
	defer func() {
		if cerr := client.Close(); cerr != nil {
			if options.Logger != nil {
				options.Logger.Warn("failed to close vertex image client",
					slog.String("error", cerr.Error()))
			} else if options.Verbose {
				fmt.Printf("failed to close vertex image client: %v\n", cerr)
			}
		}
	}()

	if options.Logger != nil {
		options.Logger.Debug("Using experimental image generation model",
			slog.Int("prompt_len", len(prompt)))
	} else if options.Verbose {
		fmt.Printf("Vertex AI image generation: model=%s, prompt=%s\n", GEMINI_MODEL_2_0_FLASH_EXP_IMAGE_GENERATION, prompt)
	}

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
	switch options.OutputFormat {
	case OutputFormatImagePNG:
		generationConfig.ResponseMIMEType = string(OutputFormatImagePNG)
	case OutputFormatImageJPG:
		generationConfig.ResponseMIMEType = "image/jpg"
	default:
		generationConfig.ResponseMIMEType = string(OutputFormatImagePNG)
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

func (l *vertexLlmImpl) GenerateEmbedding(text string) ([]float32, error) {
	return nil, errors.New("not supported. change to openrouter")
	// options := l.options

	// if options.ProjectID == "" {
	// 	return nil, errors.New("project id is required")
	// }

	// if options.Region == "" {
	// 	return nil, errors.New("region is required")
	// }

	// ctx := context.Background()
	// clientOptions, err := buildVertexClientOptions(options)
	// if err != nil {
	// 	return nil, err
	// }

	// client, err := genai.NewClient(ctx, options.ProjectID, options.Region, clientOptions...)
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to create genai client: %w", err)
	// }
	// defer func() {
	// 	if cerr := client.Close(); cerr != nil && options.Verbose {
	// 		fmt.Printf("failed to close vertex client: %v\n", cerr)
	// 	}
	// }()

	// // Use text embedding model
	// model := client.GenerativeModel("models/embedding-001")

	// // Generate embeddings
	// resp, err := model.EmbedContent(ctx, genai.Text(text))
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	// }

	// if len(resp.Embedding.Values) == 0 {
	// 	return nil, fmt.Errorf("no embeddings generated")
	// }

	// // Convert float64 to float32
	// embeddings := make([]float32, len(resp.Embedding.Values))
	// for i, v := range resp.Embedding.Values {
	// 	embeddings[i] = float32(v)
	// }

	// return embeddings, nil
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

func buildVertexClientOptions(options LlmOptions) ([]option.ClientOption, error) {
	if options.ProviderOptions != nil {
		if raw, ok := options.ProviderOptions["credentials_json"]; ok {
			switch value := raw.(type) {
			case string:
				if trimmed := strings.TrimSpace(value); trimmed != "" {
					return []option.ClientOption{option.WithCredentialsJSON([]byte(trimmed))}, nil
				}
			case []byte:
				if len(value) > 0 {
					return []option.ClientOption{option.WithCredentialsJSON(value)}, nil
				}
			default:
				return nil, fmt.Errorf("credentials_json provider option must be string or []byte")
			}
		}

		if raw, ok := options.ProviderOptions["credentials_file"]; ok {
			switch value := raw.(type) {
			case string:
				trimmed := strings.TrimSpace(value)
				if trimmed != "" {
					if _, err := os.Stat(trimmed); err != nil {
						return nil, fmt.Errorf("unable to access credentials file %s: %w", trimmed, err)
					}
					return []option.ClientOption{option.WithCredentialsFile(trimmed)}, nil
				}
			default:
				return nil, fmt.Errorf("credentials_file provider option must be string")
			}
		}
	}

	if jsonEnv := strings.TrimSpace(os.Getenv("VERTEXAI_CREDENTIALS_JSON")); jsonEnv != "" {
		return []option.ClientOption{option.WithCredentialsJSON([]byte(jsonEnv))}, nil
	}

	if fileEnv := strings.TrimSpace(os.Getenv("VERTEXAI_CREDENTIALS_FILE")); fileEnv != "" {
		if _, err := os.Stat(fileEnv); err != nil {
			return nil, fmt.Errorf("unable to access credentials file %s: %w", fileEnv, err)
		}
		return []option.ClientOption{option.WithCredentialsFile(fileEnv)}, nil
	}

	if adcFile := strings.TrimSpace(os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")); adcFile != "" {
		if _, err := os.Stat(adcFile); err != nil {
			return nil, fmt.Errorf("unable to access credentials file %s: %w", adcFile, err)
		}
		return []option.ClientOption{option.WithCredentialsFile(adcFile)}, nil
	}

	return nil, nil
}
