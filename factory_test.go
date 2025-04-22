package llm

import (
	"testing"
)

func TestNewModel(t *testing.T) {
	t.Run("valid options", func(t *testing.T) {
		options := ModelOptions{
			Provider:     ProviderMock,
			OutputFormat: OutputFormatText,
		}

		model, err := NewModel(options)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetProvider() != options.Provider {
			t.Errorf("expected provider: %v, got: %v", options.Provider, model.GetProvider())
		}

		if model.GetOutputFormat() != options.OutputFormat {
			t.Errorf("expected output format: %v, got: %v", options.OutputFormat, model.GetOutputFormat())
		}
	})

	t.Run("empty provider", func(t *testing.T) {
		options := ModelOptions{
			Provider:     "",
			OutputFormat: OutputFormatText,
		}

		_, err := NewModel(options)
		if err == nil {
			t.Fatal("expected error for empty provider, got nil")
		}

		expectedErrMsg := "provider is required"
		if err.Error() != expectedErrMsg {
			t.Errorf("expected error message: %q, got: %q", expectedErrMsg, err.Error())
		}
	})
}

func TestModelFactoryFunctions(t *testing.T) {
	// Save original config to restore after test
	originalConfig := config
	defer func() { config = originalConfig }()

	// Set up test config with required API keys
	config = struct {
		Debug                    bool
		OpenAiApiKey             string
		OpenAiDefaultModel       string
		GoogleGeminiApiKey       string
		GoogleGeminiDefaultModel string
		VertexAiProjectID        string
		VertexAiDefaultModel     string
		VertexAiRegion           string
		AnthropicApiKey          string
		AnthropicDefaultModel    string
	}{
		OpenAiApiKey:       "test-openai-key",
		GoogleGeminiApiKey: "test-gemini-key",
		VertexAiProjectID:  "test-vertex-project",
		AnthropicApiKey:    "test-anthropic-key",
	}

	t.Run("TextModel", func(t *testing.T) {
		model, err := TextModel(ProviderMock)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetProvider() != ProviderMock {
			t.Errorf("expected provider: %v, got: %v", ProviderMock, model.GetProvider())
		}

		if model.GetOutputFormat() != OutputFormatText {
			t.Errorf("expected output format: %v, got: %v", OutputFormatText, model.GetOutputFormat())
		}
	})

	t.Run("JSONModel", func(t *testing.T) {
		model, err := JSONModel(ProviderMock)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetProvider() != ProviderMock {
			t.Errorf("expected provider: %v, got: %v", ProviderMock, model.GetProvider())
		}

		if model.GetOutputFormat() != OutputFormatJSON {
			t.Errorf("expected output format: %v, got: %v", OutputFormatJSON, model.GetOutputFormat())
		}
	})

	t.Run("ImageModel", func(t *testing.T) {
		model, err := ImageModel(ProviderMock)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetProvider() != ProviderMock {
			t.Errorf("expected provider: %v, got: %v", ProviderMock, model.GetProvider())
		}

		if model.GetOutputFormat() != OutputFormatImagePNG {
			t.Errorf("expected output format: %v, got: %v", OutputFormatImagePNG, model.GetOutputFormat())
		}
	})
}

func TestCreateProvider(t *testing.T) {
	// Save original config to restore after test
	originalConfig := config
	defer func() { config = originalConfig }()

	t.Run("provides correct provider-specific defaults", func(t *testing.T) {
		// Set up test config
		config = struct {
			Debug                    bool
			OpenAiApiKey             string
			OpenAiDefaultModel       string
			GoogleGeminiApiKey       string
			GoogleGeminiDefaultModel string
			VertexAiProjectID        string
			VertexAiDefaultModel     string
			VertexAiRegion           string
			AnthropicApiKey          string
			AnthropicDefaultModel    string
		}{
			OpenAiApiKey:             "test-openai-key",
			OpenAiDefaultModel:       "gpt-4",
			GoogleGeminiApiKey:       "test-gemini-key",
			GoogleGeminiDefaultModel: "gemini-pro",
			VertexAiProjectID:        "test-vertex-project",
			VertexAiDefaultModel:     "text-bison",
			AnthropicApiKey:          "test-anthropic-key",
			AnthropicDefaultModel:    "claude-2",
		}

		testCases := []struct {
			name              string
			provider          Provider
			expectedModel     string
			expectedMaxTokens int
		}{
			{
				name:              "OpenAI provider",
				provider:          ProviderOpenAI,
				expectedModel:     "gpt-4",
				expectedMaxTokens: 4096,
			},
			{
				name:              "Gemini provider",
				provider:          ProviderGemini,
				expectedModel:     "gemini-pro",
				expectedMaxTokens: 4096,
			},
			{
				name:              "Vertex provider",
				provider:          ProviderVertex,
				expectedModel:     "text-bison",
				expectedMaxTokens: 8192,
			},
			{
				name:              "Anthropic provider",
				provider:          ProviderAnthropic,
				expectedModel:     "claude-2",
				expectedMaxTokens: 4096,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				model, err := createProvider(tc.provider, OutputFormatText)
				if err != nil {
					t.Fatalf("expected no error, got: %v", err)
				}

				if model == nil {
					t.Fatal("expected model not to be nil")
				}

				if model.GetProvider() != tc.provider {
					t.Errorf("expected provider: %v, got: %v", tc.provider, model.GetProvider())
				}

				if model.GetModel() != tc.expectedModel {
					t.Errorf("expected model: %v, got: %v", tc.expectedModel, model.GetModel())
				}

				if model.GetMaxTokens() != tc.expectedMaxTokens {
					t.Errorf("expected max tokens: %v, got: %v", tc.expectedMaxTokens, model.GetMaxTokens())
				}
			})
		}
	})

	t.Run("validates API key requirements", func(t *testing.T) {
		// Set up test config with empty API keys
		config = struct {
			Debug                    bool
			OpenAiApiKey             string
			OpenAiDefaultModel       string
			GoogleGeminiApiKey       string
			GoogleGeminiDefaultModel string
			VertexAiProjectID        string
			VertexAiDefaultModel     string
			VertexAiRegion           string
			AnthropicApiKey          string
			AnthropicDefaultModel    string
		}{}

		testCases := []struct {
			name           string
			provider       Provider
			expectedErrMsg string
		}{
			{
				name:           "Gemini without API key",
				provider:       ProviderGemini,
				expectedErrMsg: "google Gemini API key is required",
			},
			{
				name:           "Vertex without project ID",
				provider:       ProviderVertex,
				expectedErrMsg: "vertex AI project ID is required",
			},
			{
				name:           "Anthropic without API key",
				provider:       ProviderAnthropic,
				expectedErrMsg: "anthropic API key is required",
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := createProvider(tc.provider, OutputFormatText)
				if err == nil {
					t.Fatalf("expected error for %v without API key, got nil", tc.provider)
				}

				if err.Error() != tc.expectedErrMsg {
					t.Errorf("expected error message: %q, got: %q", tc.expectedErrMsg, err.Error())
				}
			})
		}
	})

	t.Run("unsupported provider", func(t *testing.T) {
		unsupportedProvider := Provider("unsupported")
		_, err := createProvider(unsupportedProvider, OutputFormatText)
		if err == nil {
			t.Fatal("expected error for unsupported provider, got nil")
		}

		expectedErrMsg := "unsupported provider: unsupported"
		if err.Error() != expectedErrMsg {
			t.Errorf("expected error message: %q, got: %q", expectedErrMsg, err.Error())
		}
	})
}
