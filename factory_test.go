package llm

import (
	"context"
	"errors"
	"testing"
)

// TestProviderRegistry tests the provider registration and factory system
func TestProviderRegistry(t *testing.T) {
	// Clear existing providers for this test
	providerMu.Lock()
	originalProviders := providerFactories
	providerFactories = make(map[Provider]LlmFactory)
	providerMu.Unlock()
	defer func() {
		// Restore original providers after test
		providerMu.Lock()
		providerFactories = originalProviders
		providerMu.Unlock()
	}()

	// Register a test provider
	testProvider := Provider("test-provider")
	RegisterProvider(testProvider, func(options LlmOptions) (LlmInterface, error) {
		return newMockImplementation(options)
	})

	// Check if provider was registered
	providerMu.RLock()
	_, exists := providerFactories[testProvider]
	providerMu.RUnlock()
	if !exists {
		t.Errorf("Provider was not registered correctly")
	}

	// Create LLM with the test provider
	llm, err := NewLLM(LlmOptions{Provider: testProvider})
	if err != nil {
		t.Errorf("Failed to create LLM with test provider: %v", err)
	}
	if llm == nil {
		t.Errorf("Created LLM is nil")
	}

	// Try to create LLM with non-existent provider
	_, err = NewLLM(LlmOptions{Provider: "non-existent"})
	if err == nil {
		t.Errorf("Expected error when creating LLM with non-existent provider, got nil")
	}
}

// TestMockLLM tests the mock LLM implementation
func TestMockLLM(t *testing.T) {
	mockLLM, _ := newMockImplementation(LlmOptions{
		MockResponse: "mock response",
	})

	// Test Generate
	response, err := mockLLM.Generate("system prompt", "test message")
	if err != nil {
		t.Errorf("Mock LLM Generate failed: %v", err)
	}
	if response != "mock response" {
		t.Errorf("Mock LLM returned unexpected response: %s", response)
	}

	// Test GenerateText
	textResponse, err := mockLLM.GenerateText("system prompt", "test message")
	if err != nil {
		t.Errorf("Mock LLM GenerateText failed: %v", err)
	}
	if textResponse != "mock response" {
		t.Errorf("Mock LLM returned unexpected text response: %s", textResponse)
	}

	// Test GenerateJSON
	jsonResponse, err := mockLLM.GenerateJSON("system prompt", "test message")
	if err != nil {
		t.Errorf("Mock LLM GenerateJSON failed: %v", err)
	}
	if jsonResponse != "mock response" {
		t.Errorf("Mock LLM returned unexpected JSON response: %s", jsonResponse)
	}

	// Test per-call MockResponse override
	overrideResponse, err := mockLLM.Generate("system prompt", "test message", LlmOptions{
		MockResponse: "override response",
	})
	if err != nil {
		t.Errorf("Mock LLM Generate with override failed: %v", err)
	}
	if overrideResponse != "override response" {
		t.Errorf("Mock LLM did not honor per-call MockResponse: %s", overrideResponse)
	}

	// Test GenerateImage
	_, err = mockLLM.GenerateImage("test prompt")
	if err != nil {
		t.Errorf("Mock LLM GenerateImage failed: %v", err)
	}

	// Test empty user message returns empty (using mock without default MockResponse)
	emptyMock, _ := newMockImplementation(LlmOptions{})
	emptyResponse, err := emptyMock.Generate("system prompt", "")
	if err != nil {
		t.Errorf("Mock LLM Generate with empty message failed: %v", err)
	}
	if emptyResponse != "" {
		t.Errorf("Mock LLM should return empty for empty user message, got: %s", emptyResponse)
	}
}

// TestLLMFactory tests the LLM factory functions
func TestLLMFactory(t *testing.T) {
	// Test CreateMockLLM
	mockLLM, err := TextModel(ProviderMock, LlmOptions{})
	if err != nil {
		t.Errorf("CreateMockLLM failed: %v", err)
	}
	if mockLLM == nil {
		t.Errorf("CreateMockLLM returned nil")
	}

	// Test with various output formats
	formats := []OutputFormat{
		OutputFormatText,
		OutputFormatJSON,
		OutputFormatImagePNG,
		OutputFormatImageJPG,
	}

	for _, format := range formats {
		mockLLM, err := TextModel(ProviderMock, LlmOptions{})
		if err != nil {
			t.Errorf("CreateMockLLM failed with format %s: %v", format, err)
		}
		if mockLLM == nil {
			t.Errorf("CreateMockLLM returned nil with format %s", format)
		}
	}
}

// CustomTestLLM is a custom LLM implementation for testing
type CustomTestLLM struct {
	generateFunc func(string, string, LlmOptions) (string, error)
	baseOptions  LlmOptions
}

func (c *CustomTestLLM) Generate(systemPrompt, userMessage string, opts ...LlmOptions) (string, error) {
	options := LlmOptions{}
	if len(opts) > 0 {
		options = opts[0]
	}
	// Merge baseOptions with override options, like real implementations
	merged := mergeOptions(c.baseOptions, options)
	if c.generateFunc != nil {
		return c.generateFunc(systemPrompt, userMessage, merged)
	}
	return "Custom test response", nil
}

func (c *CustomTestLLM) GenerateText(systemPrompt, userPrompt string, opts ...LlmOptions) (string, error) {
	options := LlmOptions{}
	if len(opts) > 0 {
		options = opts[0]
	}
	options.OutputFormat = OutputFormatText
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *CustomTestLLM) GenerateJSON(systemPrompt, userPrompt string, opts ...LlmOptions) (string, error) {
	options := LlmOptions{}
	if len(opts) > 0 {
		options = opts[0]
	}
	options.OutputFormat = OutputFormatJSON
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *CustomTestLLM) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	return []byte("test image data"), nil
}

func (c *CustomTestLLM) GenerateEmbedding(text string) ([]float32, error) {
	return nil, errors.New("not supported. change to openrouter")
}

// TestCustomProvider tests adding and using a custom provider
func TestCustomProvider(t *testing.T) {
	// Register a custom provider
	customProvider := Provider("custom-test")
	RegisterProvider(customProvider, func(options LlmOptions) (LlmInterface, error) {
		return &CustomTestLLM{
			generateFunc: func(systemPrompt, userMessage string, options LlmOptions) (string, error) {
				if systemPrompt == "error" {
					return "", errors.New("test error")
				}
				return "Custom response: " + userMessage, nil
			},
			baseOptions: options,
		}, nil
	})

	// Create LLM with the custom provider
	llm, err := NewLLM(LlmOptions{Provider: customProvider})
	if err != nil {
		t.Errorf("Failed to create LLM with custom provider: %v", err)
	}
	if llm == nil {
		t.Fatalf("Failed to create LLM with custom provider: got nil instance")
	}

	// Test successful generation
	response, err := llm.Generate("test", "hello world")
	if err != nil {
		t.Errorf("Custom LLM Generate failed: %v", err)
	}
	if response != "Custom response: hello world" {
		t.Errorf("Custom LLM returned unexpected response: %s", response)
	}

	// Test error case
	_, err = llm.Generate("error", "test")
	if err == nil {
		t.Errorf("Expected error from custom LLM, got nil")
	}
}

// TestOptionsMerging tests that options are correctly merged
func TestOptionsMerging(t *testing.T) {
	// Create a custom provider that checks options
	customProvider := Provider("options-test")
	RegisterProvider(customProvider, func(options LlmOptions) (LlmInterface, error) {
		return &CustomTestLLM{
			generateFunc: func(systemPrompt, userMessage string, options LlmOptions) (string, error) {
				// Return options as a string for testing
				if options.MaxTokens != 1000 {
					return "", errors.New("MaxTokens not set correctly")
				}
				if options.Temperature == nil || *options.Temperature != 0.5 {
					return "", errors.New("Temperature not set correctly")
				}
				if options.Model != "test-model" {
					return "", errors.New("Model not set correctly")
				}
				return "Options correct", nil
			},
			baseOptions: options,
		}, nil
	})

	// Create LLM with base options
	llm, err := NewLLM(LlmOptions{
		Provider:    customProvider,
		MaxTokens:   500,             // This should be overridden
		Temperature: PtrFloat64(0.5), // This should be used
		Model:       "default-model", // This should be overridden
	})
	if err != nil {
		t.Fatalf("Failed to create LLM for options test: %v", err)
	}
	if llm == nil {
		t.Fatalf("Failed to create LLM for options test: got nil instance")
	}

	// Call with overriding options
	response, err := llm.Generate("test", "test", LlmOptions{
		MaxTokens: 1000,         // Override base option
		Model:     "test-model", // Override base option
		// Temperature not specified, should use base option
	})

	if err != nil {
		t.Errorf("Options merging test failed: %v", err)
	}
	if response != "Options correct" {
		t.Errorf("Options were not merged correctly")
	}
}

// TestOutputFormats tests that output formats are correctly handled
func TestOutputFormats(t *testing.T) {
	mockLLM, _ := newMockImplementation(LlmOptions{})

	// Test text format
	_, err := mockLLM.GenerateText("test", "test", LlmOptions{})
	if err != nil {
		t.Errorf("GenerateText failed: %v", err)
	}

	// Test JSON format
	_, err = mockLLM.GenerateJSON("test", "test", LlmOptions{})
	if err != nil {
		t.Errorf("GenerateJSON failed: %v", err)
	}

	// Test image format
	_, err = mockLLM.GenerateImage("test", LlmOptions{OutputFormat: OutputFormatImagePNG})
	if err != nil {
		t.Errorf("GenerateImage failed: %v", err)
	}
}

// TestMockError tests that the mock implementation returns MockError when set.
func TestMockError(t *testing.T) {
	sentinel := errors.New("api down")

	// Client-level MockError
	mockLLM, _ := newMockImplementation(LlmOptions{
		MockError: sentinel,
	})

	_, err := mockLLM.Generate("sys", "user")
	if !errors.Is(err, sentinel) {
		t.Errorf("client-level MockError not returned: got %v, want %v", err, sentinel)
	}

	_, err = mockLLM.GenerateJSON("sys", "user")
	if !errors.Is(err, sentinel) {
		t.Errorf("client-level MockError not returned via GenerateJSON: got %v", err)
	}

	_, err = mockLLM.GenerateImage("prompt")
	if !errors.Is(err, sentinel) {
		t.Errorf("client-level MockError not returned via GenerateImage: got %v", err)
	}

	// Per-call MockError overrides client-level MockResponse
	mockLLM2, _ := newMockImplementation(LlmOptions{
		MockResponse: "ok",
	})
	_, err = mockLLM2.Generate("sys", "user", LlmOptions{MockError: sentinel})
	if !errors.Is(err, sentinel) {
		t.Errorf("per-call MockError did not take precedence: got %v", err)
	}

	// Per-call MockError overrides client-level MockError
	other := errors.New("different")
	mockLLM3, _ := newMockImplementation(LlmOptions{
		MockError: sentinel,
	})
	_, err = mockLLM3.Generate("sys", "user", LlmOptions{MockError: other})
	if !errors.Is(err, other) {
		t.Errorf("per-call MockError did not override client-level: got %v", err)
	}
}

// TestMockCalls tests call recording via MockInterface.
func TestMockCalls(t *testing.T) {
	raw, _ := newMockImplementation(LlmOptions{
		MockResponse: "resp",
	})
	mockLLM := raw.(MockInterface)

	// No calls initially
	calls := mockLLM.Calls()
	if len(calls) != 0 {
		t.Fatalf("expected 0 calls, got %d", len(calls))
	}

	_, _ = mockLLM.Generate("sys1", "user1")
	_, _ = mockLLM.GenerateText("sys2", "user2")
	_, _ = mockLLM.GenerateJSON("sys3", "user3")

	calls = mockLLM.Calls()
	if len(calls) != 3 {
		t.Fatalf("expected 3 calls, got %d", len(calls))
	}

	if calls[0].Method != "Generate" || calls[0].SystemPrompt != "sys1" || calls[0].UserPrompt != "user1" {
		t.Errorf("call 0 mismatch: %+v", calls[0])
	}
	if calls[1].Method != "Generate" || calls[1].SystemPrompt != "sys2" {
		t.Errorf("call 1 (GenerateText) should record Generate method: %+v", calls[1])
	}
	if calls[2].Method != "Generate" || calls[2].SystemPrompt != "sys3" {
		t.Errorf("call 2 (GenerateJSON) should record Generate method: %+v", calls[2])
	}

	// Reset clears calls
	mockLLM.Reset()
	if len(mockLLM.Calls()) != 0 {
		t.Errorf("Reset did not clear calls")
	}
}

// TestMockInterfaceAssertion tests that the mock implementation satisfies MockInterface.
func TestMockInterfaceAssertion(t *testing.T) {
	mockLLM, _ := newMockImplementation(LlmOptions{})

	if _, ok := mockLLM.(MockInterface); !ok {
		t.Errorf("mock implementation does not satisfy MockInterface")
	}
}

// TestContextPropagation tests that Context in LlmOptions is merged correctly.
func TestContextPropagation(t *testing.T) {
	ctx := context.Background()

	merged := mergeOptions(LlmOptions{}, LlmOptions{Context: ctx})
	if merged.Context != ctx {
		t.Errorf("Context not merged from per-call options")
	}

	// Client-level context preserved when per-call is nil
	merged = mergeOptions(LlmOptions{Context: ctx}, LlmOptions{})
	if merged.Context != ctx {
		t.Errorf("client-level Context not preserved")
	}
}

// TestDisableResponseFormatMerge tests that DisableResponseFormat merges correctly.
func TestDisableResponseFormatMerge(t *testing.T) {
	// Per-call true overrides
	merged := mergeOptions(LlmOptions{}, LlmOptions{DisableResponseFormat: true})
	if !merged.DisableResponseFormat {
		t.Errorf("DisableResponseFormat not merged from per-call")
	}

	// Client-level true preserved
	merged = mergeOptions(LlmOptions{DisableResponseFormat: true}, LlmOptions{})
	if !merged.DisableResponseFormat {
		t.Errorf("client-level DisableResponseFormat not preserved")
	}
}

// TestMockErrorMerge tests that MockError merges correctly.
func TestMockErrorMerge(t *testing.T) {
	sentinel := errors.New("err")

	merged := mergeOptions(LlmOptions{}, LlmOptions{MockError: sentinel})
	if merged.MockError != sentinel {
		t.Errorf("MockError not merged from per-call")
	}

	merged = mergeOptions(LlmOptions{MockError: sentinel}, LlmOptions{})
	if merged.MockError != sentinel {
		t.Errorf("client-level MockError not preserved")
	}
}
