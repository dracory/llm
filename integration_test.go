package llm

import (
	"os"
	"strings"
	"testing"
)

// skipIfNoAPIKey skips the test if the specified API key environment variable is not set
func skipIfNoAPIKey(t *testing.T, envVar string) {
	if os.Getenv(envVar) == "" {
		t.Skipf("Skipping test because %s is not set", envVar)
	}
}

// skipIfCIEnvironment skips the test if running in a CI environment
func skipIfCIEnvironment(t *testing.T) {
	if os.Getenv("CI") != "" || os.Getenv("GITHUB_ACTIONS") != "" {
		t.Skip("Skipping integration test in CI environment")
	}
}

// TestOpenAIIntegration tests the OpenAI implementation with real API calls
func TestOpenAIIntegration(t *testing.T) {
	skipIfCIEnvironment(t)
	skipIfNoAPIKey(t, "OPENAI_API_KEY")

	// Create OpenAI LLM
	llmEngine, err := TextModel(ProviderOpenAI)
	if err != nil {
		t.Fatalf("Failed to create OpenAI LLM: %v", err)
	}

	// Test text generation
	response, err := llmEngine.GenerateText(
		"You are a helpful assistant. Keep your response very short.",
		"What is a contract?",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("OpenAI text generation failed: %v", err)
	}
	if response == "" {
		t.Errorf("OpenAI returned empty response")
	}
	t.Logf("OpenAI response: %s", response)

	// Test JSON generation
	jsonResponse, err := llmEngine.GenerateJSON(
		"You are a helpful assistant. Respond with valid JSON only.",
		"Return a JSON object with the definition of a contract.",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("OpenAI JSON generation failed: %v", err)
	}
	if !strings.Contains(jsonResponse, "{") || !strings.Contains(jsonResponse, "}") {
		t.Errorf("OpenAI did not return valid JSON: %s", jsonResponse)
	}
	t.Logf("OpenAI JSON response: %s", jsonResponse)
}

// TestGeminiIntegration tests the Gemini implementation with real API calls
func TestGeminiIntegration(t *testing.T) {
	skipIfCIEnvironment(t)
	skipIfNoAPIKey(t, "GEMINI_API_KEY")

	// Create Gemini LLM
	llmEngine, err := TextModel(ProviderGemini)
	if err != nil {
		t.Fatalf("Failed to create Gemini LLM: %v", err)
	}

	// Test text generation
	response, err := llmEngine.GenerateText(
		"You are a helpful assistant. Keep your response very short.",
		"What is a contract?",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Gemini text generation failed: %v", err)
	}
	if response == "" {
		t.Errorf("Gemini returned empty response")
	}
	t.Logf("Gemini response: %s", response)

	// Test JSON generation
	jsonResponse, err := llmEngine.GenerateJSON(
		"You are a helpful assistant. Respond with valid JSON only.",
		"Return a JSON object with the definition of a contract.",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Gemini JSON generation failed: %v", err)
	}
	if !strings.Contains(jsonResponse, "{") || !strings.Contains(jsonResponse, "}") {
		t.Errorf("Gemini did not return valid JSON: %s", jsonResponse)
	}
	t.Logf("Gemini JSON response: %s", jsonResponse)
}

// TestVertexIntegration tests the Vertex implementation with real API calls
func TestVertexIntegration(t *testing.T) {
	skipIfCIEnvironment(t)
	// Skip if credentials file doesn't exist
	if _, err := os.Stat("vertexapicredentials.json"); os.IsNotExist(err) {
		t.Skip("Skipping Vertex test because vertexapicredentials.json doesn't exist")
	}

	// Create Vertex LLM
	llmEngine, err := TextModel(ProviderVertex)
	if err != nil {
		t.Fatalf("Failed to create Vertex LLM: %v", err)
	}

	// Test text generation
	response, err := llmEngine.GenerateText(
		"You are a helpful assistant. Keep your response very short.",
		"What is a contract?",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Vertex text generation failed: %v", err)
	}
	if response == "" {
		t.Errorf("Vertex returned empty response")
	}
	t.Logf("Vertex response: %s", response)

	// Test JSON generation
	jsonResponse, err := llmEngine.GenerateJSON(
		"You are a helpful assistant. Respond with valid JSON only.",
		"Return a JSON object with the definition of a contract.",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Vertex JSON generation failed: %v", err)
	}
	if !strings.Contains(jsonResponse, "{") || !strings.Contains(jsonResponse, "}") {
		t.Errorf("Vertex did not return valid JSON: %s", jsonResponse)
	}
	t.Logf("Vertex JSON response: %s", jsonResponse)
}

// TestAnthropicIntegration tests the Anthropic implementation with real API calls
func TestAnthropicIntegration(t *testing.T) {
	skipIfCIEnvironment(t)
	skipIfNoAPIKey(t, "ANTHROPIC_API_KEY")

	// Create Anthropic LLM
	llmEngine, err := TextModel(ProviderAnthropic)

	if err != nil {
		t.Fatalf("Failed to create Anthropic LLM: %v", err)
	}

	// Test text generation
	response, err := llmEngine.GenerateText(
		"You are a helpful assistant. Keep your response very short.",
		"What is a contract?",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Anthropic text generation failed: %v", err)
	}
	if response == "" {
		t.Errorf("Anthropic returned empty response")
	}
	t.Logf("Anthropic response: %s", response)

	// Test JSON generation
	jsonResponse, err := llmEngine.GenerateJSON(
		"You are a helpful assistant. Respond with valid JSON only.",
		"Return a JSON object with the definition of a contract.",
		LlmOptions{MaxTokens: 100},
	)

	if err != nil {
		t.Errorf("Anthropic JSON generation failed: %v", err)
	}
	if !strings.Contains(jsonResponse, "{") || !strings.Contains(jsonResponse, "}") {
		t.Errorf("Anthropic did not return valid JSON: %s", jsonResponse)
	}
	t.Logf("Anthropic JSON response: %s", jsonResponse)
}

// TestFactoryIntegration tests the factory pattern with real providers
func TestFactoryIntegration(t *testing.T) {
	skipIfCIEnvironment(t)

	// Test each provider if the API key is available
	providers := []struct {
		name      string
		provider  Provider
		envVar    string
		fileCheck string
	}{
		{"OpenAI", ProviderOpenAI, "OPENAI_API_KEY", ""},
		{"Gemini", ProviderGemini, "GEMINI_API_KEY", ""},
		{"Vertex", ProviderVertex, "", "vertexapicredentials.json"},
		{"Anthropic", ProviderAnthropic, "ANTHROPIC_API_KEY", ""},
		{"Mock", ProviderMock, "", ""},
	}

	for _, p := range providers {
		t.Run(string(p.name), func(t *testing.T) {
			// Skip if API key is not available
			if p.envVar != "" && os.Getenv(p.envVar) == "" {
				t.Skipf("Skipping %s test because %s is not set", p.name, p.envVar)
				return
			}

			// Skip if required file doesn't exist
			if p.fileCheck != "" {
				if _, err := os.Stat(p.fileCheck); os.IsNotExist(err) {
					t.Skipf("Skipping %s test because %s doesn't exist", p.name, p.fileCheck)
					return
				}
			}

			// Create LLM using factory
			llmEngine, err := createProvider(p.provider, OutputFormatText)
			if err != nil {
				t.Fatalf("Failed to create %s LLM: %v", p.name, err)
			}

			// Test generation
			response, err := llmEngine.Generate(
				"You are a helpful assistant. Keep your response very short.",
				"What is a contract?",
				LlmOptions{MaxTokens: 100},
			)

			if err != nil {
				t.Errorf("%s generation failed: %v", p.name, err)
			}
			if response == "" && p.provider != ProviderMock {
				t.Errorf("%s returned empty response", p.name)
			}
			t.Logf("%s response: %s", p.name, response)
		})
	}
}
