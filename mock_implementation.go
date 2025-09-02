package llm

import (
	"strings"

	"github.com/samber/lo"
)

// =======================================================================
// == CONSTRUCTOR
// =======================================================================

func newMockImplementation(options LlmOptions) (LlmInterface, error) {
	// Set default model if not provided
	if options.Model == "" {
		options.Model = "mock-model"
	}
	return &mockImplementation{
		options: options,
	}, nil
}

// =======================================================================
// == TYPE
// =======================================================================

// mockImplementation implements LlmInterface for Mock provider
type mockImplementation struct {
	options LlmOptions
}

// =======================================================================
// == IMPLEMENTATION
// =======================================================================

func (c *mockImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	options := lo.FirstOr(opts, LlmOptions{})

	// Return mock response if provided in options
	if options.MockResponse != "" {
		return options.MockResponse, nil
	}

	// Or use the one from the client options
	if c.options.MockResponse != "" {
		return c.options.MockResponse, nil
	}

	// Handle empty input
	if userMessage == "" {
		return "", nil
	}

	// Handle markdown conversion requests
	if strings.Contains(systemPrompt, "You are an expert document formatter and Markdown conversion specialist") {
		// Return a simple markdown version of the input
		return "# Test Contract\n\nThis is a test contract.\n\n1. First term\n2. Second term", nil
	}

	if strings.Contains(systemPrompt, `find the details of the contract`) {
		return `{"area_of_law": "Family Law","contract_type": "Agreement","country": "US","state": "Idaho"}`,
			nil
	}

	if strings.Contains(systemPrompt, `Your primary task is to thoroughly review`) {
		return `{
			"overallSummary": "This is a mock summary from the mock implementation.",
			"overallRisk": "Low",
			"findings": [
				{
					"section": "Term and Termination",
					"issue": "The automatic renewal clause does not specify the notice period required for termination before renewal.",
					"severity": "moderate"
				}
			],
			"recommendations": [
				"It is recommended to negotiate a shorter automatic renewal period with a clear notice period for termination."
			]
		}`,
			nil
	}

	if strings.Contains(systemPrompt, `analyze this section thoroughly`) {
		return `{"findings":[{"section":"Term and Termination","issue":"The automatic renewal clause does not specify the notice period required for termination before renewal.","severity":"moderate"},{"section":"Confidentiality","issue":"The definition of 'Confidential Information' is overly broad and could encompass publicly known information.","severity":"low"},{"section":"Payment","issue":"The late payment penalty clause does not specify the calculation method for the penalty.","severity":"high"}],"recommendations":["The contract is generally favorable, but some revisions are recommended to address the identified issues in the 'Term and Termination', 'Confidentiality', and 'Payment' sections.","It is recommended to negotiate a shorter automatic renewal period with a clear notice period for termination and refine the definition of 'Confidential Information' to exclude publicly known information.","Additionally, the late payment penalty clause should be clarified to specify the calculation method for the penalty."]}`,
			nil
	}

	return "", nil
}

func (c *mockImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatText
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *mockImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatJSON
	return c.Generate(systemPrompt, userPrompt, options)
}

func (c *mockImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})
	options.OutputFormat = OutputFormatImagePNG
	return nil, nil
}
