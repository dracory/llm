package llm

// mergeOptions merges the provided options with the default options
// and returns the merged options.
// This allows the user to override the default options.
func mergeOptions(oldOptions LlmOptions, newOptions LlmOptions) LlmOptions {
	options := LlmOptions{}
	options.Model = oldOptions.Model
	options.MaxTokens = oldOptions.MaxTokens
	options.ProviderOptions = oldOptions.ProviderOptions
	options.ProjectID = oldOptions.ProjectID
	options.Region = oldOptions.Region
	options.Temperature = oldOptions.Temperature
	options.Verbose = oldOptions.Verbose
	options.OutputFormat = oldOptions.OutputFormat

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

	if newOptions.Verbose {
		options.Verbose = newOptions.Verbose
	}

	if newOptions.OutputFormat != "" {
		options.OutputFormat = newOptions.OutputFormat
	}

	if newOptions.ProviderOptions != nil {
		options.ProviderOptions = newOptions.ProviderOptions
	}

	return options
}
