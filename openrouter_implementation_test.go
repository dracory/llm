package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDecisionModel_RequiresKey(t *testing.T) {
	if _, err := DecisionModel(ProviderOpenRouter, LlmOptions{}); err == nil {
		t.Fatal("expected error without API key")
	}
}

func TestDecide_Noul(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/alpha/decisions" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("missing bearer auth")
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"result": map[string]any{
				"answers": map[string]any{
					"is_profane": map[string]any{"noul": 0.91},
				},
			},
		})
	}))
	defer srv.Close()

	d, err := DecisionModel(ProviderOpenRouter, LlmOptions{
		ApiKey:          "test-key",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}

	answers, err := d.Decide(
		map[string]any{"text": "a55h0le"},
		map[string]Question{
			"is_profane": {
				Type:         QuestionNoul,
				Instructions: "Does the text contain disguised profanity?",
				Criteria: map[string]string{
					"true":  "Phonetic or look-alike profanity",
					"false": "Ordinary text",
				},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if a := answers["is_profane"]; a.Noul < 0.9 {
		t.Fatalf("expected noul >= 0.9, got %v", a.Noul)
	}
	if got["model"] != OPENROUTER_MODEL_JEV_1_13 {
		t.Errorf("expected default model %q, got %v", OPENROUTER_MODEL_JEV_1_13, got["model"])
	}
}

func TestDecide_TopLevelAnswersEnvelope(t *testing.T) {
	// Some deployments return answers at top level rather than under
	// result.answers — the client tolerates both.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"answers": map[string]any{
				"route": map[string]any{
					"choice":        "support",
					"probabilities": map[string]any{"support": 0.8, "sales": 0.2},
				},
			},
		})
	}))
	defer srv.Close()

	d, err := DecisionModel(ProviderOpenRouter, LlmOptions{
		ApiKey:          "test-key",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	answers, err := d.Decide(map[string]any{"text": "refund please"}, map[string]Question{
		"route": {
			Type:         QuestionChoice,
			Instructions: "Which team should handle this?",
			Options:      []string{"support", "sales"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answers["route"].Choice != "support" {
		t.Fatalf("expected choice support, got %q", answers["route"].Choice)
	}
	if answers["route"].Probabilities["support"] < 0.7 {
		t.Fatalf("expected support probability, got %v", answers["route"].Probabilities)
	}
}

// TestGenerateWithTools verifies the tools request wire format and that
// tool_calls are parsed back into ToolResponse, including the assistant
// Message needed to continue the conversation.
func TestGenerateWithTools(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []map[string]any{{
						"id":   "call_1",
						"type": "function",
						"function": map[string]any{
							"name":      "get_weather",
							"arguments": `{"city":"London"}`,
						},
					}},
				},
				"finish_reason": "tool_calls",
			}},
		})
	}))
	defer srv.Close()

	engine, err := NewLLM(LlmOptions{
		Provider:        ProviderOpenRouter,
		ApiKey:          "test-key",
		Model:           "test-model",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	toolsLLM, ok := engine.(ToolsInterface)
	if !ok {
		t.Fatal("openrouter engine does not implement ToolsInterface")
	}

	resp, err := toolsLLM.GenerateWithTools("sys", "weather?", LlmOptions{
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Request shape: tools array carries the function definition.
	tools, ok := got["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected one tool in request, got %v", got["tools"])
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Fatalf("expected tool name get_weather, got %v", fn["name"])
	}
	msgs := got["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected system+user messages, got %d", len(msgs))
	}

	// Response shape: tool call parsed, assistant message reusable.
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(resp.ToolCalls))
	}
	call := resp.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "get_weather" || call.Arguments != `{"city":"London"}` {
		t.Fatalf("unexpected tool call %+v", call)
	}
	if resp.Message.Role != RoleAssistant || len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected assistant message with tool call, got %+v", resp.Message)
	}
}

// TestGenerateWithTools_Continuation verifies tool results are sent back
// as tool messages with matching tool_call_id.
func TestGenerateWithTools_Continuation(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"role": "assistant", "content": "It is 12C."},
				"finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	engine, _ := NewLLM(LlmOptions{
		Provider:        ProviderOpenRouter,
		ApiKey:          "test-key",
		Model:           "test-model",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	toolsLLM := engine.(ToolsInterface)

	resp, err := toolsLLM.GenerateWithTools("sys", "", LlmOptions{
		Tools: []Tool{{Name: "get_weather"}},
		Messages: []Message{
			{Role: RoleUser, Content: "weather?"},
			{
				Role:      RoleAssistant,
				ToolCalls: []ToolCall{{ID: "call_1", Name: "get_weather", Arguments: `{"city":"London"}`}},
			},
			{Role: RoleTool, ToolCallID: "call_1", Name: "get_weather", Content: `{"temp":"12C"}`},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "It is 12C." {
		t.Fatalf("unexpected content %q", resp.Content)
	}

	msgs := got["messages"].([]any)
	if len(msgs) != 4 { // system + user + assistant + tool
		t.Fatalf("expected 4 messages, got %d: %v", len(msgs), msgs)
	}
	assistant := msgs[2].(map[string]any)
	if assistant["role"] != RoleAssistant || assistant["tool_calls"] == nil {
		t.Fatalf("expected assistant tool_calls, got %v", assistant)
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != RoleTool || tool["tool_call_id"] != "call_1" {
		t.Fatalf("expected tool result message, got %v", tool)
	}
}

func TestDecide_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte("bad"))
	}))
	defer srv.Close()

	d, _ := DecisionModel(ProviderOpenRouter, LlmOptions{
		ApiKey:          "test-key",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	if _, err := d.Decide(map[string]any{}, map[string]Question{}); err == nil {
		t.Fatal("expected error on non-200")
	}
}

func TestToOpenAITools_ServerTool(t *testing.T) {
	tools := []Tool{
		{
			Type: "openrouter:web_search",
		},
		{
			Name:        "get_stock_price",
			Description: "Get stock price",
		},
	}

	converted := toOpenAITools(tools)
	if len(converted) != 2 {
		t.Fatalf("expected 2 converted tools, got %d", len(converted))
	}

	if string(converted[0].Type) != "openrouter:web_search" {
		t.Errorf("expected tool[0] type 'openrouter:web_search', got %q", converted[0].Type)
	}
	if converted[0].Function != nil {
		t.Errorf("expected tool[0] function to be nil, got %+v", converted[0].Function)
	}

	if string(converted[1].Type) != "function" {
		t.Errorf("expected tool[1] type 'function', got %q", converted[1].Type)
	}
	if converted[1].Function == nil || converted[1].Function.Name != "get_stock_price" {
		t.Errorf("unexpected tool[1] function %+v", converted[1].Function)
	}
}

func TestGenerateWithTools_ServerTool(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"role": "assistant", "content": "Search results..."},
				"finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	engine, err := NewLLM(LlmOptions{
		Provider:        ProviderOpenRouter,
		ApiKey:          "test-key",
		Model:           "test-model",
		ProviderOptions: map[string]any{"base_url": srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	toolsLLM := engine.(ToolsInterface)

	resp, err := toolsLLM.GenerateWithTools("sys", "search for latest Go release", LlmOptions{
		Tools: []Tool{
			{Type: "openrouter:web_search"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Search results..." {
		t.Fatalf("unexpected content %q", resp.Content)
	}

	tools, ok := got["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool in payload, got %v", got["tools"])
	}
	t0 := tools[0].(map[string]any)
	if t0["type"] != "openrouter:web_search" {
		t.Fatalf("expected tool type 'openrouter:web_search', got %v", t0["type"])
	}
}
