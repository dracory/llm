package llm

import (
	"context"
	"errors"
	"testing"
)

func TestMockModel_Complete(t *testing.T) {
	ctx := context.Background()

	t.Run("default response", func(t *testing.T) {
		model := NewMockModel()
		request := CompletionRequest{
			SystemPrompt: "You are a helpful assistant",
			UserPrompt:   "Hello, world!",
			MaxTokens:    10,
			Temperature:  0.7,
		}

		response, err := model.Complete(ctx, request)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		expected := "This is a mock response"
		if response.Text != expected {
			t.Errorf("expected response text: %q, got: %q", expected, response.Text)
		}

		if response.TokensUsed != 5 {
			t.Errorf("expected tokens used: 5, got: %d", response.TokensUsed)
		}
	})

	t.Run("empty prompt", func(t *testing.T) {
		model := NewMockModel()
		request := CompletionRequest{
			SystemPrompt: "",
			UserPrompt:   "",
			MaxTokens:    10,
			Temperature:  0.7,
		}

		_, err := model.Complete(ctx, request)
		if !errors.Is(err, ErrInvalidRequest) {
			t.Fatalf("expected ErrInvalidRequest, got: %v", err)
		}
	})

	t.Run("predefined error", func(t *testing.T) {
		expectedErr := errors.New("mock error")
		model := &MockModel{
			Error: expectedErr,
			options: ModelOptions{
				Provider:     ProviderMock,
				OutputFormat: OutputFormatText,
			},
		}
		request := CompletionRequest{
			SystemPrompt: "You are a helpful assistant",
			UserPrompt:   "Hello, world!",
			MaxTokens:    10,
			Temperature:  0.7,
		}

		_, err := model.Complete(ctx, request)
		if !errors.Is(err, expectedErr) {
			t.Fatalf("expected error: %v, got: %v", expectedErr, err)
		}
	})

	t.Run("echo response", func(t *testing.T) {
		model := &MockModel{
			Response: CompletionResponse{
				Text:       "",
				TokensUsed: 0,
			},
			options: ModelOptions{
				Provider:     ProviderMock,
				OutputFormat: OutputFormatText,
			},
		}
		request := CompletionRequest{
			SystemPrompt: "You are a helpful assistant",
			UserPrompt:   "Hello, world!",
			MaxTokens:    10,
			Temperature:  0.7,
		}

		response, err := model.Complete(ctx, request)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		expected := "Echo: You are a helpful assistant\n\nHello, world!"
		if response.Text != expected {
			t.Errorf("expected response text: %q, got: %q", expected, response.Text)
		}
	})
}

func TestMockModel_GetterMethods(t *testing.T) {
	model := NewMockModel()

	t.Run("GetProvider", func(t *testing.T) {
		if model.GetProvider() != ProviderMock {
			t.Errorf("expected provider: %v, got: %v", ProviderMock, model.GetProvider())
		}
	})

	t.Run("GetOutputFormat", func(t *testing.T) {
		if model.GetOutputFormat() != OutputFormatText {
			t.Errorf("expected output format: %v, got: %v", OutputFormatText, model.GetOutputFormat())
		}
	})

	t.Run("GetModel", func(t *testing.T) {
		expected := "mock-model"
		if model.GetModel() != expected {
			t.Errorf("expected model: %v, got: %v", expected, model.GetModel())
		}
	})

	t.Run("GetMaxTokens", func(t *testing.T) {
		expected := 4096
		if model.GetMaxTokens() != expected {
			t.Errorf("expected max tokens: %v, got: %v", expected, model.GetMaxTokens())
		}
	})

	t.Run("GetTemperature", func(t *testing.T) {
		expected := 0.7
		if model.GetTemperature() != expected {
			t.Errorf("expected temperature: %v, got: %v", expected, model.GetTemperature())
		}
	})
}

func TestNewMockModelWithOptions(t *testing.T) {
	options := ModelOptions{
		Provider:     ProviderGemini, // This should be overridden to ProviderMock
		OutputFormat: OutputFormatJSON,
		Model:        "custom-model",
		MaxTokens:    1000,
		Temperature:  0.5,
		ApiKey:       "test-api-key",
		ProjectID:    "test-project",
		Region:       "test-region",
		Verbose:      true,
	}

	model := NewMockModelWithOptions(options)

	// Provider should always be ProviderMock
	if model.GetProvider() != ProviderMock {
		t.Errorf("expected provider to be ProviderMock, got: %v", model.GetProvider())
	}

	// Other options should be respected
	if model.GetOutputFormat() != options.OutputFormat {
		t.Errorf("expected output format: %v, got: %v", options.OutputFormat, model.GetOutputFormat())
	}

	if model.GetModel() != options.Model {
		t.Errorf("expected model: %v, got: %v", options.Model, model.GetModel())
	}

	if model.GetMaxTokens() != options.MaxTokens {
		t.Errorf("expected max tokens: %v, got: %v", options.MaxTokens, model.GetMaxTokens())
	}

	if model.GetTemperature() != options.Temperature {
		t.Errorf("expected temperature: %v, got: %v", options.Temperature, model.GetTemperature())
	}

	if model.GetApiKey() != options.ApiKey {
		t.Errorf("expected API key: %v, got: %v", options.ApiKey, model.GetApiKey())
	}

	if model.GetProjectID() != options.ProjectID {
		t.Errorf("expected project ID: %v, got: %v", options.ProjectID, model.GetProjectID())
	}

	if model.GetRegion() != options.Region {
		t.Errorf("expected region: %v, got: %v", options.Region, model.GetRegion())
	}

	if model.GetVerbose() != options.Verbose {
		t.Errorf("expected verbose: %v, got: %v", options.Verbose, model.GetVerbose())
	}
}
