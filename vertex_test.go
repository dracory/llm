package llm

import (
	"testing"
)

func TestNewVertexModel(t *testing.T) {
	t.Run("with project ID", func(t *testing.T) {
		options := ModelOptions{
			Provider:  ProviderVertex,
			ProjectID: "test-project",
		}

		model, err := newVertexModel(options)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetProjectID() != options.ProjectID {
			t.Errorf("expected project ID: %v, got: %v", options.ProjectID, model.GetProjectID())
		}

		if model.GetRegion() != "us-central1" {
			t.Errorf("expected default region to be us-central1, got: %v", model.GetRegion())
		}

		if model.GetModel() != VertexModelGemini20FlashLite {
			t.Errorf("expected default model to be %v, got: %v", VertexModelGemini20FlashLite, model.GetModel())
		}
	})

	t.Run("with custom region and model", func(t *testing.T) {
		customRegion := "europe-west4"
		customModel := VertexModelGemini25ProPreview
		options := ModelOptions{
			Provider:  ProviderVertex,
			ProjectID: "test-project",
			Region:    customRegion,
			Model:     customModel,
		}

		model, err := newVertexModel(options)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		if model == nil {
			t.Fatal("expected model not to be nil")
		}

		if model.GetRegion() != customRegion {
			t.Errorf("expected region: %v, got: %v", customRegion, model.GetRegion())
		}

		if model.GetModel() != customModel {
			t.Errorf("expected model: %v, got: %v", customModel, model.GetModel())
		}
	})

	t.Run("without project ID", func(t *testing.T) {
		options := ModelOptions{
			Provider: ProviderVertex,
			Region:   "us-central1",
		}

		_, err := newVertexModel(options)
		if err == nil {
			t.Fatal("expected error for missing project ID, got nil")
		}

		expectedErrMsg := "vertex AI project ID is required"
		if err.Error() != expectedErrMsg {
			t.Errorf("expected error message: %q, got: %q", expectedErrMsg, err.Error())
		}
	})
}

func TestFindVertexModelName(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty",
			input:    "",
			expected: VertexModelGemini20FlashLite,
		},
		{
			name:     "pro-preview",
			input:    "some-pro-preview-model",
			expected: VertexModelGemini25ProPreview,
		},
		{
			name:     "pro",
			input:    "gemini-pro",
			expected: VertexModelGemini15Pro,
		},
		{
			name:     "flash-lite",
			input:    "gemini-flash-lite",
			expected: VertexModelGemini20FlashLite,
		},
		{
			name:     "flash",
			input:    "gemini-flash",
			expected: VertexModelGemini20Flash,
		},
		{
			name:     "custom model",
			input:    "custom-model",
			expected: "custom-model",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := findVertexModelName(tc.input)
			if result != tc.expected {
				t.Errorf("expected: %v, got: %v", tc.expected, result)
			}
		})
	}
}

// Integration test that would need to be run with real credentials
// This is commented out as it would require actual API calls
/*
func TestVertexImplementation_Complete_Integration(t *testing.T) {
	// Skip if not in integration test mode
	projectID := os.Getenv("VERTEX_PROJECT_ID")
	if projectID == "" || os.Getenv("RUN_INTEGRATION_TESTS") != "true" {
		t.Skip("Skipping integration test (no project ID or not in integration test mode)")
	}

	ctx := context.Background()

	options := ModelOptions{
		Provider:     ProviderVertex,
		ProjectID:    projectID,
		Region:       "us-central1",
		Model:        VertexModelGemini20FlashLite,
		MaxTokens:    100,
		Temperature:  0.7,
		OutputFormat: OutputFormatText,
	}

	model, err := newVertexModel(options)
	if err != nil {
		t.Fatalf("Error creating Vertex model: %v", err)
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
