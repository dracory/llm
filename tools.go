package llm

import (
	"github.com/sashabaranov/go-openai"
)

// toOpenAITools converts provider-agnostic tool definitions to the
// go-openai request representation.
func toOpenAITools(tools []Tool) []openai.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]openai.Tool, len(tools))
	for i, t := range tools {
		toolType := t.Type
		if toolType == "" {
			toolType = string(openai.ToolTypeFunction)
		}

		if toolType == string(openai.ToolTypeFunction) {
			params := t.Parameters
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			out[i] = openai.Tool{
				Type: openai.ToolTypeFunction,
				Function: &openai.FunctionDefinition{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  params,
					Strict:      t.Strict,
				},
			}
		} else {
			tool := openai.Tool{
				Type: openai.ToolType(toolType),
			}
			if t.Name != "" || t.Description != "" || t.Parameters != nil {
				tool.Function = &openai.FunctionDefinition{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				}
			}
			out[i] = tool
		}
	}
	return out
}

// toOpenAIToolChoice maps the ToolChoice option to the go-openai field
// type. Empty/"auto" return nil so the provider default applies.
func toOpenAIToolChoice(choice string) any {
	switch choice {
	case "", ToolChoiceAuto:
		return nil
	case ToolChoiceNone, ToolChoiceRequired:
		return choice
	default:
		// Treat any other value as a specific function name.
		return openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: choice},
		}
	}
}

// toOpenAIMessages builds the OpenAI-style message list: system prompt
// first (when non-empty), then the caller-supplied conversation, then the
// user prompt (when non-empty).
func toOpenAIMessages(systemPrompt string, messages []Message, userPrompt string) []openai.ChatCompletionMessage {
	out := make([]openai.ChatCompletionMessage, 0, len(messages)+2)
	if systemPrompt != "" {
		out = append(out, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}
	for _, m := range messages {
		out = append(out, toOpenAIMessage(m))
	}
	if userPrompt != "" {
		out = append(out, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userPrompt,
		})
	}
	return out
}

// toOpenAIMessage converts a single conversation message. Assistant
// messages may carry ToolCalls; tool results carry ToolCallID + Name.
func toOpenAIMessage(m Message) openai.ChatCompletionMessage {
	out := openai.ChatCompletionMessage{
		Role:       m.Role,
		Content:    m.Content,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
	}
	for _, tc := range m.ToolCalls {
		tcType := tc.Type
		if tcType == "" {
			tcType = string(openai.ToolTypeFunction)
		}
		out.ToolCalls = append(out.ToolCalls, openai.ToolCall{
			ID:   tc.ID,
			Type: openai.ToolType(tcType),
			Function: openai.FunctionCall{
				Name:      tc.Name,
				Arguments: tc.Arguments,
			},
		})
	}
	return out
}

// fromOpenAIChoice converts the first response choice into a
// ToolResponse, including the assistant Message needed to continue the
// conversation on a follow-up call.
func fromOpenAIChoice(msg openai.ChatCompletionMessage) ToolResponse {
	resp := ToolResponse{
		Content: msg.Content,
		Message: Message{
			Role:    RoleAssistant,
			Content: msg.Content,
		},
	}
	for _, tc := range msg.ToolCalls {
		tcType := string(tc.Type)
		if tcType == "" {
			tcType = string(openai.ToolTypeFunction)
		}
		call := ToolCall{
			ID:        tc.ID,
			Type:      tcType,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		}
		resp.ToolCalls = append(resp.ToolCalls, call)
		resp.Message.ToolCalls = append(resp.Message.ToolCalls, call)
	}
	return resp
}
