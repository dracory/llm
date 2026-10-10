package llm

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// LlmInterface is an interface for making LLM API calls
type LlmInterface interface {
	// GenerateText generates a text response from the LLM based on the given prompt
	GenerateText(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

	// GenerateJSON generates a JSON response from the LLM based on the given prompt
	GenerateJSON(systemPrompt string, userPrompt string, options ...LlmOptions) (string, error)

	// GenerateImage generates an image from the LLM based on the given prompt
	GenerateImage(prompt string, options ...LlmOptions) ([]byte, error)

	// DEPRECATED: Generate generates a response from the LLM based on the given prompt and options
	Generate(systemPrompt string, userMessage string, options ...LlmOptions) (string, error)

	// GenerateEmbedding generates embeddings for the given text
	GenerateEmbedding(text string) ([]float32, error)

	// Decide answers typed questions about a state object with
	// calibrated probabilities (System One decision models — see the
	// DECISION MODELS block below). Providers without a decisions
	// endpoint return an error. Use DecisionModel (factory.go) to
	// construct one directly.
	Decide(state map[string]any, questions map[string]Question, options ...LlmOptions) (map[string]Answer, error)
}

// == DECISION MODELS =========================================================
//
// Decision models (System One) are NOT generative: instead of producing
// text they answer typed questions about a state object and return
// calibrated probabilities. OpenRouter hosts a whole family of them on
// the shared Decisions contract (POST /alpha/decisions or the
// /v1/systemone API — the request/response shape is the same):
//
//   typesafe/jev-1.13               64K ctx   $0.042/M in, output free
//   perplexity/decider-1.1-27b     262K ctx   $0.02/M in  (cheapest, ≤128 q/call)
//   nace-ai/drex-v1.5              128K ctx   $0.04/M in  (open weights)
//   upstage/solar-decide-flash     524K ctx   $0.05/M in
//   microsoft/microsoft-decision-1  33K ctx   $0.042/M in
//   cloudflare/clef-omni            66K ctx   $0.15/M in  (multimodal: images)
//
// They are not drop-in chat replacements — they replace the
// prompt-and-parse step for routing, classification, verification,
// moderation, and rubric grading. The model is chosen with
// LlmOptions.Model (default OPENROUTER_MODEL_JEV_1_13 — slugs in
// openrouter_models.go).

// Question type primitives supported by the Decisions contract.
const (
	// QuestionNoul — "does this condition hold?" → probability of yes.
	QuestionNoul = "noul"
	// QuestionChoice — "which of these options?" → option + probabilities.
	QuestionChoice = "choice"
	// QuestionScore — "where on this ordered scale?" → weighted position.
	QuestionScore = "score"
)

// Question is one typed decision question keyed by name in the request.
type Question struct {
	// Type is one of QuestionNoul/QuestionChoice/QuestionScore.
	Type string `json:"type"`
	// Instructions is the natural-language question.
	Instructions string `json:"instructions"`
	// Criteria gives plain-language definitions per outcome — for noul
	// the keys are "true"/"false"; tuning criteria usually beats tuning
	// thresholds.
	Criteria map[string]string `json:"criteria,omitempty"`
	// Options lists the allowed answers for QuestionChoice.
	Options []string `json:"options,omitempty"`
	// Scale lists the ordered labels for QuestionScore.
	Scale []string `json:"scale,omitempty"`
}

// Answer is one question's typed result.
type Answer struct {
	// Noul is the probability-of-yes for QuestionNoul.
	Noul float64 `json:"noul,omitempty"`
	// Choice is the selected option for QuestionChoice.
	Choice string `json:"choice,omitempty"`
	// Score is the probability-weighted position for QuestionScore.
	Score float64 `json:"score,omitempty"`
	// Probabilities carries per-option/per-level probabilities where the
	// API returns them.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Confidence is the model's self-reported confidence.
	Confidence float64 `json:"confidence,omitempty"`
}

type LlmOptions struct {
	// Provider specifies which LLM provider to use
	Provider Provider

	// MockResponse, if not empty, will be returned by the mock implementation
	// instead of making an actual API call. This is useful for testing.
	MockResponse string `json:"-"`

	// MockError, if not nil, will be returned by the mock implementation
	// instead of making an actual API call. This is useful for testing
	// failure paths. MockError takes precedence over MockResponse.
	MockError error `json:"-"`

	// Context, if not nil, will be used for the API call, allowing callers
	// to cancel in-flight requests or apply deadlines. When nil,
	// context.Background() is used.
	Context context.Context `json:"-"`

	// Timeout, if greater than zero, sets the HTTP client timeout for
	// providers that construct their own client. When zero, each provider's
	// default is used (30s for Anthropic, Custom and Gemini; no timeout for
	// the OpenAI-compatible providers).
	Timeout time.Duration

	// DisableResponseFormat, when true, prevents the provider from sending
	// a response_format parameter. Useful for providers/models that reject
	// structured-output requests (e.g. some OpenRouter routes for
	// GenerateJSON).
	DisableResponseFormat bool

	// ApiKey specifies the API key for the LLM provider
	ApiKey string

	// ProjectID specifies the project ID for the LLM (used by Vertex AI)
	ProjectID string

	// Region specifies the region for the LLM (used by Vertex AI)
	Region string

	// Model specifies the LLM model to use
	Model string

	// MaxTokens specifies the maximum number of tokens to generate
	MaxTokens int

	// Temperature controls the randomness of the response.
	// A higher temperature (e.g., 0.8) makes the output more random and creative,
	// while a lower temperature (e.g., 0.2) makes the output more focused and deterministic.
	// Use PtrFloat64(0.7) to set, or leave nil to use the provider default.
	Temperature *float64

	// Verbose controls whether to log detailed information
	Verbose bool

	// Logger specifies a logger to use for error logging
	Logger *slog.Logger

	// OutputFormat specifies the output format from the LLM
	OutputFormat OutputFormat

	// Additional options specific to the LLM provider
	ProviderOptions map[string]any
}

// MockCall records a single call made to the mock implementation.
type MockCall struct {
	Method       string // e.g. "Generate", "GenerateText", "GenerateJSON"
	SystemPrompt string
	UserPrompt   string
	Options      LlmOptions
}

// MockInterface is implemented by the mock provider. It extends LlmInterface
// with introspection so tests can assert which calls were made. Obtain it via
// a type assertion on the LlmInterface returned by the factory:
//
//	mock, ok := engine.(llm.MockInterface)
//	if ok { _ = mock.Calls() }
type MockInterface interface {
	LlmInterface

	// Calls returns a copy of all recorded calls in invocation order.
	Calls() []MockCall

	// Reset clears the recorded call history.
	Reset()
}

// LlmFactory is a function type that creates a new LLM instance
// Now returns (LlmInterface, error)
type LlmFactory func(options LlmOptions) (LlmInterface, error)

var (
	// providerMu protects providerFactories from concurrent access
	providerMu sync.RWMutex
	// providerFactories maps provider names to their factory functions
	providerFactories = make(map[Provider]LlmFactory)
)

// RegisterProvider registers a new LLM provider factory
func RegisterProvider(provider Provider, factory LlmFactory) {
	providerMu.Lock()
	defer providerMu.Unlock()
	providerFactories[provider] = factory
}

// RegisterCustomProvider registers a custom LLM provider
func RegisterCustomProvider(name string, factory LlmFactory) {
	RegisterProvider(Provider(name), factory)
}

// NewLLM creates a new LLM instance based on the provider specified in options
func NewLLM(options LlmOptions) (LlmInterface, error) {
	if options.Provider == "" {
		// Default to OpenAI if no provider is specified
		options.Provider = ProviderOpenAI
	}

	providerMu.RLock()
	factory, exists := providerFactories[options.Provider]
	providerMu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("unsupported LLM provider: %s", options.Provider)
	}

	llm, err := factory(options)
	if err != nil {
		return nil, err
	}
	return llm, nil
}

// PtrFloat64 returns a pointer to the given float64 value.
// This is a convenience helper for setting Temperature in LlmOptions.
func PtrFloat64(v float64) *float64 {
	return &v
}

// init registers the built-in LLM providers
func init() {
	// Register built-in providers
	RegisterProvider(ProviderOpenAI, func(options LlmOptions) (LlmInterface, error) {
		return newOpenaiImplementation(options)
	})

	RegisterProvider(ProviderGemini, func(options LlmOptions) (LlmInterface, error) {
		return newGeminiImplementation(options)
	})

	RegisterProvider(ProviderVertex, func(options LlmOptions) (LlmInterface, error) {
		return newVertexImplementation(options)
	})

	RegisterProvider(ProviderMock, func(options LlmOptions) (LlmInterface, error) {
		return newMockImplementation(options)
	})

	RegisterProvider(ProviderAnthropic, func(options LlmOptions) (LlmInterface, error) {
		return newAnthropicImplementation(options)
	})

	RegisterProvider(ProviderOpenRouter, func(options LlmOptions) (LlmInterface, error) {
		return newOpenRouterImplementation(options)
	})

	RegisterProvider(ProviderCustom, func(options LlmOptions) (LlmInterface, error) {
		return newCustomImplementation(options)
	})
}
