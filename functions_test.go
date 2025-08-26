package llm

import (
	"reflect"
	"testing"
)

func TestMergeOptions(t *testing.T) {
	// Define base options
	baseOptions := LlmOptions{
		Model:           "base-model",
		MaxTokens:       100,
		ProviderOptions: map[string]any{"baseKey": "baseValue"},
		ProjectID:       "base-project",
		Region:          "base-region",
		Temperature:     0.5,
		Verbose:         false,
		OutputFormat:    OutputFormatText,
	}

	// Define test cases
	testCases := []struct {
		name        string
		oldOptions  LlmOptions
		newOptions  LlmOptions
		wantOptions LlmOptions
	}{
		{
			name:       "No new options",
			oldOptions: baseOptions,
			newOptions: LlmOptions{},
			wantOptions: LlmOptions{
				Model:           "base-model",
				MaxTokens:       100,
				ProviderOptions: map[string]any{"baseKey": "baseValue"},
				ProjectID:       "base-project",
				Region:          "base-region",
				Temperature:     0.5,
				Verbose:         false,
				OutputFormat:    OutputFormatText,
			},
		},
		{
			name:       "Override some options",
			oldOptions: baseOptions,
			newOptions: LlmOptions{
				Model:     "new-model",
				MaxTokens: 200,
				Verbose:   true,
			},
			wantOptions: LlmOptions{
				Model:           "new-model", // Overridden
				MaxTokens:       200,         // Overridden
				ProviderOptions: map[string]any{"baseKey": "baseValue"},
				ProjectID:       "base-project",
				Region:          "base-region",
				Temperature:     0.5,
				Verbose:         true, // Overridden
				OutputFormat:    OutputFormatText,
			},
		},
		{
			name:       "Override all options",
			oldOptions: baseOptions,
			newOptions: LlmOptions{
				Model:           "new-model-all",
				MaxTokens:       500,
				ProviderOptions: map[string]any{"newKey": "newValue"},
				ProjectID:       "new-project",
				Region:          "new-region",
				Temperature:     0.9,
				Verbose:         true,
				OutputFormat:    OutputFormatJSON,
			},
			wantOptions: LlmOptions{
				Model:           "new-model-all",
				MaxTokens:       500,
				ProviderOptions: map[string]any{"newKey": "newValue"},
				ProjectID:       "new-project",
				Region:          "new-region",
				Temperature:     0.9,
				Verbose:         true,
				OutputFormat:    OutputFormatJSON,
			},
		},
		{
			name:       "Override only ProviderOptions",
			oldOptions: baseOptions,
			newOptions: LlmOptions{
				ProviderOptions: map[string]any{"anotherKey": 123},
			},
			wantOptions: LlmOptions{
				Model:           "base-model",
				MaxTokens:       100,
				ProviderOptions: map[string]any{"anotherKey": 123}, // Overridden
				ProjectID:       "base-project",
				Region:          "base-region",
				Temperature:     0.5,
				Verbose:         false,
				OutputFormat:    OutputFormatText,
			},
		},
		{
			name:       "Override with zero values (except bool)",
			oldOptions: baseOptions,
			newOptions: LlmOptions{
				Model:       "",    // Should not override
				MaxTokens:   0,     // Should not override
				ProjectID:   "",    // Should not override
				Region:      "",    // Should not override
				Temperature: 0,     // Should not override
				Verbose:     false, // Should not change from base false
			},
			wantOptions: LlmOptions{
				Model:           "base-model",
				MaxTokens:       100,
				ProviderOptions: map[string]any{"baseKey": "baseValue"},
				ProjectID:       "base-project",
				Region:          "base-region",
				Temperature:     0.5,
				Verbose:         false,
				OutputFormat:    OutputFormatText,
			},
		},
		{
			name: "Override Verbose from true to true",
			oldOptions: LlmOptions{
				Verbose: true, // Start with true
			},
			newOptions: LlmOptions{
				Verbose: true, // Override with true
			},
			wantOptions: LlmOptions{
				Verbose: true, // Should remain true
			},
		},
		{
			name: "Override Verbose from true to false (should not happen)",
			oldOptions: LlmOptions{
				Verbose: true, // Start with true
			},
			newOptions: LlmOptions{
				Verbose: false, // Override with false - current logic only sets true
			},
			wantOptions: LlmOptions{
				Verbose: true, // Stays true because `if newOptions.Verbose` requires true
			},
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotOptions := mergeOptions(tc.oldOptions, tc.newOptions)
			if !reflect.DeepEqual(gotOptions, tc.wantOptions) {
				t.Errorf("mergeOptions() = %+v, want %+v", gotOptions, tc.wantOptions)
			}
		})
	}
}
