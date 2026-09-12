package examples_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/dracory/llm"
)

// Example_mockBasic shows the simplest usage: create a mock LLM and get a
// canned response. No API key, no network — ideal for unit tests.
func Example_mockBasic() {
	engine, err := llm.TextModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "Hello from the mock!",
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := engine.GenerateText("You are a helpful assistant.", "Say hello")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp)
	// Output: Hello from the mock!
}

// Example_mockJSON shows how to use GenerateJSON with a mock that returns
// a JSON string. This is the pattern used for structured-output tasks like
// classification or extraction.
func Example_mockJSON() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: `{"risk_tier":"low","domain":"general","reasoning":"safe content"}`,
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := engine.GenerateJSON(
		"You are a risk classifier. Respond with JSON only.",
		"This is a simple blog post about cooking.",
	)
	if err != nil {
		log.Fatal(err)
	}

	var result struct {
		RiskTier  string `json:"risk_tier"`
		Domain    string `json:"domain"`
		Reasoning string `json:"reasoning"`
	}
	if err := json.Unmarshal([]byte(resp), &result); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("tier=%s domain=%s\n", result.RiskTier, result.Domain)
	// Output: tier=low domain=general
}

// Example_mockError shows how to simulate an API failure using MockError.
// This is useful for testing retry/failure paths without a real API.
func Example_mockError() {
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockError: errors.New("rate limit exceeded"),
	})
	if err != nil {
		log.Fatal(err)
	}

	_, err = engine.GenerateJSON("system", "user")
	fmt.Println(err)
	// Output: rate limit exceeded
}

// Example_mockPerCallOverride shows how per-call options override the
// client-level options. Here the client returns "default" but a per-call
// MockResponse changes the output for that single call.
func Example_mockPerCallOverride() {
	engine, err := llm.TextModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "default response",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Without override — uses client-level MockResponse
	r1, _ := engine.GenerateText("sys", "user")
	fmt.Println("default:", r1)

	// With per-call override
	r2, _ := engine.GenerateText("sys", "user", llm.LlmOptions{
		MockResponse: "per-call override",
	})
	fmt.Println("override:", r2)
	// Output:
	// default: default response
	// override: per-call override
}

// Example_mockCallRecording shows how to use MockInterface to inspect
// which calls were made. Type-assert the engine to MockInterface to get
// access to Calls() and Reset().
func Example_mockCallRecording() {
	engine, err := llm.TextModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "ok",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Type-assert to MockInterface for introspection
	mock, ok := engine.(llm.MockInterface)
	if !ok {
		log.Fatal("engine does not implement MockInterface")
	}

	// Make some calls
	_, _ = mock.GenerateText("sys prompt A", "user message A")
	_, _ = mock.GenerateJSON("sys prompt B", "user message B")

	// Inspect recorded calls
	calls := mock.Calls()
	for i, c := range calls {
		fmt.Printf("call %d: method=%s system=%q user=%q\n", i, c.Method, c.SystemPrompt, c.UserPrompt)
	}

	// Clear history
	mock.Reset()
	fmt.Println("after reset:", len(mock.Calls()))
	// Output:
	// call 0: method=Generate system="sys prompt A" user="user message A"
	// call 1: method=Generate system="sys prompt B" user="user message B"
	// after reset: 0
}
