package llm

import (
	"testing"
)

func TestNewOpenAIModel(t *testing.T) {
	t.Run("with api key", func(t *testing.T) {
		options := ModelOptions{
			Provider: ProviderOpenAI,
			ApiKey:   "test-api-key",
		}

		model, err := newOpenAIModel(options)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetApiKey() != options.ApiKey {
			t.Errorf("expected API key: %v, got: %v", options.ApiKey, model.GetApiKey())
		}

		if model.GetModel() != OpenAIModelGPT4Turbo {
			t.Errorf("expected default model to be %v, got: %v", OpenAIModelGPT4Turbo, model.GetModel())
		}
	})

	t.Run("with custom model", func(t *testing.T) {
		customModel := OpenAIModelGPT35Turbo
		options := ModelOptions{
			Provider: ProviderOpenAI,
			ApiKey:   "test-api-key",
			Model:    customModel,
		}

		model, err := newOpenAIModel(options)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetModel() != customModel {
			t.Errorf("expected model: %v, got: %v", customModel, model.GetModel())
		}
	})

	t.Run("without api key", func(t *testing.T) {
		options := ModelOptions{
			Provider: ProviderOpenAI,
			ApiKey:   "",
		}

		_, err := newOpenAIModel(options)
		if err == nil {
			t.Fatal("expected error for missing API key, got nil")
		}

		expectedErrMsg := "openai API key not provided"
		if err.Error() != expectedErrMsg {
			t.Errorf("expected error message: %q, got: %q", expectedErrMsg, err.Error())
		}
	})
}

// Integration test that would need to be run with a real API key
// This is commented out as it would require actual API calls
/*
func TestOpenAIImplementation_Complete_Integration(t *testing.T) {
	// Skip if no API key available or if not in integration test mode
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" || os.Getenv("RUN_INTEGRATION_TESTS") != "true" {
		t.Skip("Skipping integration test (no API key or not in integration test mode)")
	}

	ctx := context.Background()

	options := ModelOptions{
		Provider:     ProviderOpenAI,
		ApiKey:       apiKey,
		Model:        OpenAIModelGPT35Turbo,
		MaxTokens:    100,
		Temperature:  0.7,
		OutputFormat: OutputFormatText,
	}

	model, err := newOpenAIModel(options)
	if err != nil {
		t.Fatalf("Error creating OpenAI model: %v", err)
	}

	request := CompletionRequest{
		SystemPrompt: "You are a helpful assistant. Keep your answers brief.",
		UserPrompt:   "What is the capital of France?",
	}

	response, err := model.Complete(ctx, request)
	if err != nil {
		t.Fatalf("Error in Complete: %v", err)
	}

	if response.Text == "" {
		t.Error("Expected non-empty response")
	}

	if response.TokensUsed <= 0 {
		t.Errorf("Expected non-zero tokens used, got: %d", response.TokensUsed)
	}

	t.Logf("Response: %s", response.Text)
	t.Logf("Tokens used: %d", response.TokensUsed)
}
*/
