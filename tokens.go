package llm

import (
	"strings"
)

// CountTokens provides a simple approximation of token counting
// Note: This is a basic implementation and not accurate for all models
// Production code should use model-specific tokenizers
func CountTokens(text string) int {
	if text == "" {
		return 0
	}

	// Count words
	words := strings.Fields(text)
	tokenCount := len(words)

	// Count punctuation
	for _, char := range text {
		if strings.ContainsRune(".,!?;:", char) {
			tokenCount++
		}
	}

	return tokenCount
}

// EstimateMaxTokens estimates the maximum number of tokens that could be generated
// given the model's context window size and the prompt length
func EstimateMaxTokens(promptTokens, contextWindowSize int) int {
	if promptTokens >= contextWindowSize {
		return 0
	}

	return contextWindowSize - promptTokens
}
