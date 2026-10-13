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

// Message roles used in LlmOptions.Messages for tool-enabled conversations.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// ToolChoice values accepted by LlmOptions.ToolChoice. Any other
// non-empty value is treated as a specific tool name the model must call.
const (
	// ToolChoiceAuto lets the model decide whether to call a tool (default).
	ToolChoiceAuto = "auto"
	// ToolChoiceNone forbids tool calls.
	ToolChoiceNone = "none"
	// ToolChoiceRequired forces the model to call one of the tools.
	ToolChoiceRequired = "required"
)

// Provider represents an LLM provider type
type Provider string

// Supported LLM providers
const (
	ProviderOpenAI     Provider = "openai"
	ProviderGemini     Provider = "gemini"
	ProviderVertex     Provider = "vertex"
	ProviderMock       Provider = "mock"
	ProviderAnthropic  Provider = "anthropic"
	ProviderOpenRouter Provider = "openrouter"
	ProviderCustom     Provider = "custom"
)
