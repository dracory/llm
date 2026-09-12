package examples_test

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/dracory/llm"
)

// Example_openRouterBasic shows the basic setup for OpenRouter.
// You need an API key from https://openrouter.ai/keys.
//
// This example uses the mock to avoid a real API call — in production
// swap ProviderMock for ProviderOpenRouter and set ApiKey.
func Example_openRouterBasic() {
	// --- Production setup (uncomment and set your key) ---
	// engine, err := llm.JSONModel(llm.ProviderOpenRouter, llm.LlmOptions{
	//     ApiKey:   "sk-or-v1-...",
	//     Model:    llm.OPENROUTER_MODEL_LING_3_0_FLASH,
	//     MaxTokens: 2000,
	// })

	// --- Test setup (no network) ---
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: `{"sentiment":"positive","confidence":0.95}`,
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := engine.GenerateJSON(
		"You are a sentiment classifier. Respond with JSON: {sentiment, confidence}",
		"I love this product, it works great!",
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp)
	// Output: {"sentiment":"positive","confidence":0.95}
}

// Example_openRouterCustomBaseURL shows how to override the OpenRouter
// base URL via ProviderOptions. This is useful for proxies, gateways,
// or local test servers.
func Example_openRouterCustomBaseURL() {
	// In production with a proxy:
	// engine, err := llm.JSONModel(llm.ProviderOpenRouter, llm.LlmOptions{
	//     ApiKey: "sk-or-v1-...",
	//     Model:  llm.OPENROUTER_MODEL_GPT_OSS_120B,
	//     ProviderOptions: map[string]any{
	//         "base_url": "https://my-proxy.example.com/v1",
	//     },
	// })

	// Demonstrating with mock (no network)
	engine, err := llm.TextModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "routed via custom base URL",
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, _ := engine.GenerateText("sys", "user")
	fmt.Println(resp)
	// Output: routed via custom base URL
}

// Example_contextCancellation shows how to pass a context with a timeout
// to cancel an in-flight LLM request. Set LlmOptions.Context on the
// per-call options to propagate the context to the provider.
func Example_contextCancellation() {
	engine, err := llm.TextModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse: "should not see this if cancelled",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Create a context that times out after 1 second
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Pass the context via per-call options
	resp, err := engine.GenerateText("sys", "user", llm.LlmOptions{
		Context: ctx,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("response:", resp)
	// Output: response: should not see this if cancelled
}

// Example_disableResponseFormat shows how to disable the response_format
// parameter. Some OpenRouter providers reject structured-output requests
// for certain models. Set DisableResponseFormat to omit it.
func Example_disableResponseFormat() {
	// In production with a model that doesn't support response_format:
	// engine, err := llm.JSONModel(llm.ProviderOpenRouter, llm.LlmOptions{
	//     ApiKey:              "sk-or-v1-...",
	//     Model:               "some-model-without-json-mode",
	//     DisableResponseFormat: true,
	// })

	// Demonstrating with mock
	engine, err := llm.JSONModel(llm.ProviderMock, llm.LlmOptions{
		MockResponse:          `{"result":"ok"}`,
		DisableResponseFormat: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, _ := engine.GenerateJSON("sys", "user")
	fmt.Println(resp)
	// Output: {"result":"ok"}
}

// Example_modelSelection shows the range of model constants available.
// Pick based on price/performance trade-offs for your use case.
func Example_modelSelection() {
	models := []struct {
		name      string
		slug      string
		inputCost string
	}{
		{"Ling 3.0 Flash", llm.OPENROUTER_MODEL_LING_3_0_FLASH, "$0.021/M"},
		{"DeepSeek V4 Flash 0731", llm.OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_0731, "$0.05/M"},
		{"GLM 5.3 Flash", llm.OPENROUTER_MODEL_GLM_5_3_FLASH, "$0.075/M"},
		{"Qwen3.8 Flash", llm.OPENROUTER_MODEL_QWEN_3_8_FLASH, "$0.15/M"},
		{"Gemini 3.5 Flash Lite", llm.OPENROUTER_MODEL_GEMINI_3_5_FLASH_LITE, "$0.30/M"},
		{"GPT-OSS-120B", llm.OPENROUTER_MODEL_GPT_OSS_120B, "$0.072/M"},
	}

	fmt.Println("Cheapest text models on OpenRouter:")
	for _, m := range models {
		fmt.Printf("  %-25s %s  (%s)\n", m.name, m.slug, m.inputCost)
	}
}
