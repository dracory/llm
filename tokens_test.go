package llm

import "testing"

func TestCountTokens(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected int
	}{
		{
			name:     "empty string",
			text:     "",
			expected: 0,
		},
		{
			name:     "single word",
			text:     "hello",
			expected: 1,
		},
		{
			name:     "multiple words",
			text:     "hello world",
			expected: 2,
		},
		{
			name:     "with punctuation",
			text:     "hello, world!",
			expected: 4, // "hello", ",", "world", "!"
		},
		{
			name:     "complex sentence",
			text:     "This is a test. It has multiple sentences, with various punctuation marks!",
			expected: 14, // 11 words + 3 punctuation marks
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CountTokens(tt.text)
			if result != tt.expected {
				t.Errorf("CountTokens(%q) = %d, expected %d", tt.text, result, tt.expected)
			}
		})
	}
}

func TestEstimateMaxTokens(t *testing.T) {
	tests := []struct {
		name              string
		promptTokens      int
		contextWindowSize int
		expected          int
	}{
		{
			name:              "empty prompt",
			promptTokens:      0,
			contextWindowSize: 4096,
			expected:          4096,
		},
		{
			name:              "half used context",
			promptTokens:      2048,
			contextWindowSize: 4096,
			expected:          2048,
		},
		{
			name:              "nearly full context",
			promptTokens:      4000,
			contextWindowSize: 4096,
			expected:          96,
		},
		{
			name:              "exceeding context",
			promptTokens:      5000,
			contextWindowSize: 4096,
			expected:          0,
		},
		{
			name:              "equal to context",
			promptTokens:      4096,
			contextWindowSize: 4096,
			expected:          0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EstimateMaxTokens(tt.promptTokens, tt.contextWindowSize)
			if result != tt.expected {
				t.Errorf("EstimateMaxTokens(%d, %d) = %d, expected %d",
					tt.promptTokens, tt.contextWindowSize, result, tt.expected)
			}
		})
	}
}
