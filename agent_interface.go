package llm

// AgentInterface defines the core interface that all agents must implement
type AgentInterface interface {
	// SetRole sets the role of the agent
	// i.e. "You are a helpful assistant"
	SetRole(role string)

	// GetRole returns the role of the agent
	GetRole() string

	// SetTask sets the task for the agent
	// i.e. "Your task is to write a book about self-improvement"
	SetTask(task string)

	// GetTask returns the task of the agent
	GetTask() string

	// Execute runs the agent and returns the response
	Execute() (response string, err error)
}
