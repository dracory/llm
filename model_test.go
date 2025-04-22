package llm

import (
	"context"
	"testing"
)

func TestModelImplementation_Complete(t *testing.T) {
	ctx := context.Background()

	t.Run("mock provider returns mock response", func(t *testing.T) {
		model := &modelImplementation{
			options: ModelOptions{
				Provider:     ProviderMock,
				OutputFormat: OutputFormatText,
			},
		}

		request := CompletionRequest{
			SystemPrompt: "You are a helpful assistant",
			UserPrompt:   "Test prompt",
			MaxTokens:    100,
			Temperature:  0.7,
		}

		response, err := model.Complete(ctx, request)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		expectedText := "This is a mock completion response"
		if response.Text != expectedText {
			t.Errorf("expected text: %q, got: %q", expectedText, response.Text)
		}

		expectedTokens := 7
		if response.TokensUsed != expectedTokens {
			t.Errorf("expected tokens: %d, got: %d", expectedTokens, response.TokensUsed)
		}
	})

	t.Run("unimplemented provider returns error", func(t *testing.T) {
		model := &modelImplementation{
			options: ModelOptions{
				Provider:     ProviderOpenAI,
				OutputFormat: OutputFormatText,
			},
		}

		request := CompletionRequest{
			SystemPrompt: "You are a helpful assistant",
			UserPrompt:   "Test prompt",
			MaxTokens:    100,
			Temperature:  0.7,
		}

		_, err := model.Complete(ctx, request)
		if err == nil {
			t.Fatal("expected error for unimplemented provider, got nil")
		}

		expectedErrMsg := "provider openai not yet implemented"
		if err.Error() != expectedErrMsg {
			t.Errorf("expected error message: %q, got: %q", expectedErrMsg, err.Error())
		}
	})
}

func TestModelImplementation_Getters(t *testing.T) {
	// Test options
	options := ModelOptions{
		Provider:     ProviderOpenAI,
		OutputFormat: OutputFormatJSON,
		ApiKey:       "test-api-key",
		Model:        "gpt-4",
		MaxTokens:    2048,
		Temperature:  0.7,
		ProjectID:    "test-project",
		Region:       "us-central1",
		Verbose:      true,
	}

	model := &modelImplementation{options: options}

	t.Run("GetProvider", func(t *testing.T) {
		if model.GetProvider() != options.Provider {
			t.Errorf("expected provider: %v, got: %v", options.Provider, model.GetProvider())
		}
	})

	t.Run("GetOutputFormat", func(t *testing.T) {
		if model.GetOutputFormat() != options.OutputFormat {
			t.Errorf("expected output format: %v, got: %v", options.OutputFormat, model.GetOutputFormat())
		}
	})

	t.Run("GetApiKey", func(t *testing.T) {
		if model.GetApiKey() != options.ApiKey {
			t.Errorf("expected API key: %v, got: %v", options.ApiKey, model.GetApiKey())
		}
	})

	t.Run("GetModel", func(t *testing.T) {
		if model.GetModel() != options.Model {
			t.Errorf("expected model: %v, got: %v", options.Model, model.GetModel())
		}
	})

	t.Run("GetMaxTokens", func(t *testing.T) {
		if model.GetMaxTokens() != options.MaxTokens {
			t.Errorf("expected max tokens: %v, got: %v", options.MaxTokens, model.GetMaxTokens())
		}
	})

	t.Run("GetTemperature", func(t *testing.T) {
		if model.GetTemperature() != options.Temperature {
			t.Errorf("expected temperature: %v, got: %v", options.Temperature, model.GetTemperature())
		}
	})

	t.Run("GetProjectID", func(t *testing.T) {
		if model.GetProjectID() != options.ProjectID {
			t.Errorf("expected project ID: %v, got: %v", options.ProjectID, model.GetProjectID())
		}
	})

	t.Run("GetRegion", func(t *testing.T) {
		if model.GetRegion() != options.Region {
			t.Errorf("expected region: %v, got: %v", options.Region, model.GetRegion())
		}
	})

	t.Run("GetVerbose", func(t *testing.T) {
		if model.GetVerbose() != options.Verbose {
			t.Errorf("expected verbose: %v, got: %v", options.Verbose, model.GetVerbose())
		}
	})
}
