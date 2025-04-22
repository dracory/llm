package llm

// OutputFormat specifies the desired output format from the LLM
type OutputFormat string

const (
	OutputFormatText     OutputFormat = "text"
	OutputFormatJSON     OutputFormat = "json"
	OutputFormatXML      OutputFormat = "xml"
	OutputFormatYAML     OutputFormat = "yaml"
	OutputFormatEnum     OutputFormat = "enum"
	OutputFormatImagePNG OutputFormat = "image/png"
	OutputFormatImageJPG OutputFormat = "image/jpeg"
)

// Provider represents an LLM provider type
type Provider string

// Supported LLM providers
const (
	ProviderOpenAI    Provider = "openai"
	ProviderGemini    Provider = "gemini"
	ProviderVertex    Provider = "vertex"
	ProviderMock      Provider = "mock"
	ProviderAnthropic Provider = "anthropic"
	ProviderCustom    Provider = "custom"
)
