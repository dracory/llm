package llm

// mergeOptions merges the provided options with the default options
// and returns the merged options.
// This allows the user to override the default options.
func mergeOptions(oldOptions LlmOptions, newOptions LlmOptions) LlmOptions {
	options := LlmOptions{}
	options.Provider = oldOptions.Provider
	options.ApiKey = oldOptions.ApiKey
	options.Model = oldOptions.Model
	options.MaxTokens = oldOptions.MaxTokens
	options.ProviderOptions = oldOptions.ProviderOptions
	options.ProjectID = oldOptions.ProjectID
	options.Region = oldOptions.Region
	options.Temperature = oldOptions.Temperature
	options.Verbose = oldOptions.Verbose
	options.OutputFormat = oldOptions.OutputFormat
	options.Logger = oldOptions.Logger
	options.MockResponse = oldOptions.MockResponse

	if newOptions.Provider != "" {
		options.Provider = newOptions.Provider
	}

	if newOptions.ApiKey != "" {
		options.ApiKey = newOptions.ApiKey
	}

	if newOptions.Model != "" {
		options.Model = newOptions.Model
	}

	if newOptions.MaxTokens != 0 {
		options.MaxTokens = newOptions.MaxTokens
	}

	if newOptions.ProjectID != "" {
		options.ProjectID = newOptions.ProjectID
	}

	if newOptions.Region != "" {
		options.Region = newOptions.Region
	}

	if newOptions.Temperature != 0 {
		options.Temperature = newOptions.Temperature
	}

	// Verbose can only be turned on via merge, not turned off,
	// because the zero value (false) is indistinguishable from "not set".
	if newOptions.Verbose {
		options.Verbose = true
	}

	if newOptions.OutputFormat != "" {
		options.OutputFormat = newOptions.OutputFormat
	}

	if newOptions.ProviderOptions != nil {
		options.ProviderOptions = newOptions.ProviderOptions
	}

	if newOptions.Logger != nil {
		options.Logger = newOptions.Logger
	}

	if newOptions.MockResponse != "" {
		options.MockResponse = newOptions.MockResponse
	}

	return options
}
