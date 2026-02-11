package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

// anthropicImplementation implements LlmInterface for Anthropic
type anthropicImplementation struct {
	apiKey          string
	model           string
	maxTokens       int
	temperature     float64
	verbose         bool
	logger          *slog.Logger
	providerOptions map[string]any
	httpClient      *http.Client
}

func mergeProviderOptions(base map[string]any, override map[string]any) map[string]any {
	if base == nil && override == nil {
		return nil
	}

	merged := map[string]any{}

	for k, v := range base {
		merged[k] = v
	}

	for k, v := range override {
		merged[k] = v
	}

	return merged
}

func buildAnthropicHTTPClient(providerOptions map[string]any) (*http.Client, error) {
	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS12,
	}

	rootCAFile := valueFromProviderOrEnv(providerOptions, "anthropic_root_ca_file", "ANTHROPIC_ROOT_CA_FILE")
	rootCAPEM := valueFromProviderOrEnv(providerOptions, "anthropic_root_ca_pem", "ANTHROPIC_ROOT_CA_PEM")
	spkiHash := valueFromProviderOrEnv(providerOptions, "anthropic_spki_hash", "ANTHROPIC_EXPECTED_SPKI_HASH")

	customRootCA := false
	if rootCAFile != "" || rootCAPEM != "" {
		rootPool, err := x509.SystemCertPool()
		if err != nil || rootPool == nil {
			rootPool = x509.NewCertPool()
		}

		if rootCAPEM != "" {
			if ok := rootPool.AppendCertsFromPEM([]byte(rootCAPEM)); !ok {
				return nil, fmt.Errorf("anthropic: invalid root CA PEM")
			}
			customRootCA = true
		}

		if rootCAFile != "" {
			pemBytes, err := os.ReadFile(rootCAFile)
			if err != nil {
				return nil, fmt.Errorf("anthropic: unable to read root CA file %s: %w", rootCAFile, err)
			}
			if ok := rootPool.AppendCertsFromPEM(pemBytes); !ok {
				return nil, fmt.Errorf("anthropic: invalid root CA file %s", rootCAFile)
			}
			customRootCA = true
		}

		if customRootCA {
			tlsConfig.RootCAs = rootPool
		}
	}

	spkiHash = strings.TrimSpace(spkiHash)
	spkiHash = strings.TrimPrefix(spkiHash, "sha256/")

	if spkiHash != "" {
		expectedPin, err := base64.StdEncoding.DecodeString(spkiHash)
		if err != nil {
			return nil, fmt.Errorf("anthropic: invalid SPKI hash: %w", err)
		}

		tlsConfig.VerifyConnection = func(state tls.ConnectionState) error {
			if len(state.PeerCertificates) == 0 {
				return fmt.Errorf("anthropic: no peer certificates for pinning")
			}

			leaf := state.PeerCertificates[0]
			hash := sha256.Sum256(leaf.RawSubjectPublicKeyInfo)
			if subtle.ConstantTimeCompare(hash[:], expectedPin) != 1 {
				return fmt.Errorf("anthropic: certificate pin mismatch")
			}

			return nil
		}
	}

	transport := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		TLSClientConfig:     tlsConfig,
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
	}

	return &http.Client{
		Timeout:   30 * time.Second,
		Transport: transport,
	}, nil
}

func valueFromProviderOrEnv(providerOptions map[string]any, key string, envKey string) string {
	if providerOptions != nil {
		if raw, ok := providerOptions[key]; ok {
			switch v := raw.(type) {
			case string:
				if trimmed := strings.TrimSpace(v); trimmed != "" {
					return trimmed
				}
			case []byte:
				if trimmed := strings.TrimSpace(string(v)); trimmed != "" {
					return trimmed
				}
			}
		}
	}

	return strings.TrimSpace(os.Getenv(envKey))
}

// newAnthropicImplementation creates a new Anthropic provider implementation
func newAnthropicImplementation(options LlmOptions) (LlmInterface, error) {
	model := options.Model
	if model == "" {
		model = "claude-3-opus-20240229" // Default to Claude 3 Opus
	}

	client, err := buildAnthropicHTTPClient(options.ProviderOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to configure anthropic http client: %w", err)
	}

	return &anthropicImplementation{
		apiKey:          options.ApiKey,
		model:           model,
		maxTokens:       options.MaxTokens,
		temperature:     derefFloat64(options.Temperature, 0.7),
		verbose:         options.Verbose,
		logger:          options.Logger,
		providerOptions: options.ProviderOptions,
		httpClient:      client,
	}, nil
}

// baseOptions returns the base LlmOptions from the struct fields for merging.
func (a *anthropicImplementation) baseOptions() LlmOptions {
	return LlmOptions{
		Model:           a.model,
		MaxTokens:       a.maxTokens,
		Temperature:     &a.temperature,
		Verbose:         a.verbose,
		Logger:          a.logger,
		ProviderOptions: a.providerOptions,
	}
}

// Generate implements LlmInterface
func (a *anthropicImplementation) Generate(systemPrompt string, userMessage string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	merged := mergeOptions(a.baseOptions(), perCall)

	// Validate API key
	if a.apiKey == "" {
		return "", fmt.Errorf("anthropic api key not provided")
	}

	ctx := context.Background()

	model := merged.Model
	maxTokens := merged.MaxTokens
	temperature := derefFloat64(merged.Temperature, a.temperature)

	// Prepare request body
	requestBody := map[string]interface{}{
		"model":       model,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"system":      systemPrompt,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": userMessage,
			},
		},
	}

	// Add response format if JSON is requested
	if merged.OutputFormat == OutputFormatJSON {
		requestBody["response_format"] = map[string]string{
			"type": "json_object",
		}
	}

	// Convert request body to JSON
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	// Send request
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	if resp == nil {
		return "", fmt.Errorf("failed to send request: received nil response")
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			if a.logger != nil {
				a.logger.Warn("failed to close response body",
					slog.String("error", cerr.Error()))
			} else if a.verbose {
				fmt.Printf("failed to close response body: %v\n", cerr)
			}
		}
	}()

	// Read response body (limit to 10 MB to prevent memory exhaustion)
	body, err := io.ReadAll(io.LimitReader(resp.Body, 10<<20))
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	// Check for error response
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API returned error: %s", string(body))
	}

	// Parse response
	var responseData map[string]interface{}
	if err := json.Unmarshal(body, &responseData); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	// Extract content from response
	content, ok := responseData["content"].([]interface{})
	if !ok || len(content) == 0 {
		return "", fmt.Errorf("invalid response format")
	}

	// Get text from first content item
	firstContent, ok := content[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid content format")
	}

	text, ok := firstContent["text"].(string)
	if !ok {
		return "", fmt.Errorf("invalid text format")
	}

	return strings.TrimSpace(text), nil
}

// GenerateText implements LlmInterface
func (a *anthropicImplementation) GenerateText(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatText
	return a.Generate(systemPrompt, userPrompt, perCall)
}

// GenerateJSON implements LlmInterface
func (a *anthropicImplementation) GenerateJSON(systemPrompt string, userPrompt string, opts ...LlmOptions) (string, error) {
	perCall := LlmOptions{}
	if len(opts) > 0 {
		perCall = opts[0]
	}
	perCall.OutputFormat = OutputFormatJSON
	systemPrompt += "\nYou must respond with valid JSON only. Do not include any text outside the JSON."
	return a.Generate(systemPrompt, userPrompt, perCall)
}

// GenerateImage implements LlmInterface
func (a *anthropicImplementation) GenerateImage(prompt string, opts ...LlmOptions) ([]byte, error) {
	// Note: As of now, Anthropic doesn't have a direct image generation API like DALL-E
	// options := lo.IfF(len(opts) > 0, func() LlmOptions { return opts[0] }).Else(LlmOptions{})

	// Anthropic doesn't support image generation natively
	if a.logger != nil {
		a.logger.Warn("Image generation is not supported by Anthropic API")
	} else if a.verbose {
		fmt.Println("Image generation is not supported by Anthropic API")
	}

	return nil, fmt.Errorf("image generation not supported by Anthropic")
}

// GenerateEmbedding implements LlmInterface
func (a *anthropicImplementation) GenerateEmbedding(text string) ([]float32, error) {
	return nil, errors.New("not supported. change to openrouter")
}
