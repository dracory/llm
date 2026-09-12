package examples_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/dracory/llm"
)

// triageResult is the structured output we expect from the LLM.
type triageResult struct {
	RiskTier  string   `json:"risk_tier"`
	Domain    string   `json:"domain"`
	Claims    []string `json:"claims"`
	Reasoning string   `json:"reasoning"`
}

// runTriage is a realistic LLM-backed classifier function. It takes an
// engine (any llm.LlmInterface) and content, calls GenerateJSON, parses
// the response, and returns the structured result.
//
// In production you'd use llm.ProviderOpenRouter; in tests you use
// llm.ProviderMock with a canned JSON response or MockError.
func runTriage(engine llm.LlmInterface, content string) (*triageResult, error) {
	if engine == nil {
		return nil, errors.New("no LLM engine configured")
	}

	systemPrompt := `You are a content risk classifier for an expert verification platform.
Classify the given content and respond with JSON only:
{
  "risk_tier": "low" | "medium" | "high" | "critical",
  "domain": "general" | "medical" | "legal" | "financial" | "technical",
  "claims": ["list of specific factual claims that need verification"],
  "reasoning": "brief explanation of the classification"
}`

	resp, err := engine.GenerateJSON(systemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	var result triageResult
	if err := json.Unmarshal([]byte(resp), &result); err != nil {
		return nil, fmt.Errorf("invalid JSON from LLM: %w", err)
	}

	return &result, nil
}

// Example_triageSuccess shows a successful triage classification using
// the mock provider with a canned JSON response.
func Example_triageSuccess() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: `{
			"risk_tier": "high",
			"domain": "medical",
			"claims": ["Drug X cures cancer", "FDA approved in 2024"],
			"reasoning": "Medical claims about drug efficacy require expert verification"
		}`,
	})
	if err != nil {
		log.Fatal(err)
	}

	result, err := runTriage(engine, "New study shows Drug X cures cancer, FDA approved 2024.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("tier=%s domain=%s claims=%v\n", result.RiskTier, result.Domain, result.Claims)
	// Output: tier=high domain=medical claims=[Drug X cures cancer FDA approved in 2024]
}

// Example_triageLLMError shows how to test the error path using MockError.
// The mock returns the configured error instead of making a real API call.
func Example_triageLLMError() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockError: errors.New("openrouter 503 service unavailable"),
	})
	if err != nil {
		log.Fatal(err)
	}

	_, err = runTriage(engine, "some content")
	fmt.Println(err)
	// Output: LLM call failed: openrouter 503 service unavailable
}

// Example_triageInvalidJSON shows how to test the invalid-JSON path.
// The mock returns a non-JSON string, causing json.Unmarshal to fail.
func Example_triageInvalidJSON() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "This is not JSON, just plain text.",
	})
	if err != nil {
		log.Fatal(err)
	}

	_, err = runTriage(engine, "some content")
	fmt.Println(err)
	// Output: invalid JSON from LLM: invalid character 'T' looking for beginning of value
}

// Example_triageNoEngine shows the nil-engine guard.
func Example_triageNoEngine() {
	_, err := runTriage(nil, "some content")
	fmt.Println(err)
	// Output: no LLM engine configured
}

// Example_triageWithCallRecording shows how to verify that the LLM was
// called with the correct prompts using MockInterface.Calls().
func Example_triageWithCallRecording() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: `{"risk_tier":"low","domain":"general","claims":[],"reasoning":"ok"}`,
	})
	if err != nil {
		log.Fatal(err)
	}

	content := "A simple recipe for chocolate cake."
	_, _ = runTriage(engine, content)

	// Verify the LLM was called with the right content
	mock, ok := engine.(llm.MockInterface)
	if !ok {
		log.Fatal("engine does not implement MockInterface")
	}

	calls := mock.Calls()
	if len(calls) != 1 {
		log.Fatalf("expected 1 call, got %d", len(calls))
	}

	// The user prompt should contain our content
	fmt.Println("LLM was called with content:", calls[0].UserPrompt)
	// Output: LLM was called with content: A simple recipe for chocolate cake.
}
