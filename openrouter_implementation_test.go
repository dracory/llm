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
