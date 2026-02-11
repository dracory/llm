package llm

// ===========================================================================//
// OpenAI Models
// ===========================================================================//

// OpenAI GPT-OSS-20B
// Input $0.04/M Output $0.15/M Context 131,072
const OPENROUTER_MODEL_GPT_OSS_20B = "openai/gpt-oss-20b"

// OpenAI GPT-OSS-120B
// Input $0.072/M Output $0.28/M Context 131,072
const OPENROUTER_MODEL_GPT_OSS_120B = "openai/gpt-oss-120b"

// OpenAI O4 Mini
// Input $1.10/M Output $4.40/M Context 200,000
const OPENROUTER_MODEL_O4_MINI = "openai/o4-mini"

// OpenAI GPT-4.1 Nano
// Input $0.10/M Output $0.40/M Context 1,047,576
const OPENROUTER_MODEL_GPT_4_1_NANO = "openai/gpt-4.1-nano"

// OpenAI GPT-5 Nano
// Input $0.05/M Output $0.40/M Context 400,000
const OPENROUTER_MODEL_GPT_5_NANO = "openai/gpt-5-nano"

// OpenAI GPT-5.1
// Input $1.25/M Output $10.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_1 = "openai/gpt-5.1"

// OpenAI GPT-5.2
// Input $1.75/M Output $14.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2 = "openai/gpt-5.2"

// OpenAI GPT-5.2 Chat (Instant)
// Input $1.75/M Output $14.00/M Context 128,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_CHAT = "openai/gpt-5.2-chat"

// OpenAI GPT-5.2 Pro
// Input $21.00/M Output $168.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_PRO = "openai/gpt-5.2-pro"

// OpenAI GPT-5.2 Codex
// Input $1.75/M Output $14.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_CODEX = "openai/gpt-5.2-codex"

// ===========================================================================//
// Anthropic Models
// ===========================================================================//

// Anthropic Claude Sonnet 4
// Input $3.00/M Output $15.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"

// Anthropic Claude Sonnet 4.5
// Input $3.00/M Output $15.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"

// Anthropic Claude Haiku 4.5
// Input $0.80/M Output $4.00/M Context 200,000
const OPENROUTER_MODEL_CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"

// Anthropic Claude Opus 4.5
// Input $5.00/M Output $25.00/M Context 200,000
const OPENROUTER_MODEL_CLAUDE_OPUS_4_5 = "anthropic/claude-opus-4.5"

// Anthropic Claude Opus 4.6
// Input $5.00/M Output $25.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_OPUS_4_6 = "anthropic/claude-opus-4.6"

// ===========================================================================//
// Google Models
// ===========================================================================//

// Google Gemma 3 12B
// Input $0.048/M Output $0.193/M Context 96,000
const OPENROUTER_MODEL_GEMMA_3_12B_IT = "google/gemma-3-12b-it"

// Google Gemma 3 27B
// Input $0.067/M Output $0.267/M Context 96,000
const OPENROUTER_MODEL_GEMMA_3_27B_IT = "google/gemma-3-27b-it"

// Google Gemini 2.5 Flash Lite
// Input $0.10/M Output $0.40/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"

// Google Gemini 2.5 Flash
// Input $0.30/M Output $2.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH = "google/gemini-2.5-flash"

// Google Gemini 2.5 Pro
// Input $1.25/M Output $10/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_PRO = "google/gemini-2.5-pro"

// Google Gemini 3 Flash Preview
// Input $0.50/M Output $3.00/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"

// Google Gemini 3 Pro Preview
// Input $2/M Output $12/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"

// ===========================================================================//
// Mistral Models
// ===========================================================================//

// Mistral Mistral Nemo
// Input $0.02/M Output $0.04/M Context 131,072
const OPENROUTER_MODEL_MISTRAL_NEMO = "mistralai/mistral-nemo"

// Mistral Mistral Medium 3.1
// Input $0.40/M Output $2/M Context 131,072
const OPENROUTER_MODEL_MISTRAL_MEDIUM_3_1 = "mistralai/mistral-medium-3.1"

// Mistral Devstral 2512
// Input $0.05/M Output $0.22/M Context 262,144
const OPENROUTER_MODEL_DEVSTRAL_2512 = "mistralai/devstral-2512"

// ===========================================================================//
// Qwen Models
// ===========================================================================//

// Qwen Qwen3 235B A22B Instruct 2507
// Input $0.078/M Output $0.312/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_235B_A22B_INSTRUCT_2507 = "qwen/qwen3-235b-a22b-2507"

// Qwen Qwen3 30B A3B
// Input $0.02/M Output $0.08/M Context 40,960
const OPENROUTER_MODEL_QWEN_3_30B_A3B = "qwen/qwen3-30b-a3b"

// Qwen Qwen3 Max Thinking
// Input $1.20/M Output $6.00/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_MAX_THINKING = "qwen/qwen3-max-thinking"

// Qwen Qwen3 Coder Next
// Input $0.07/M Output $0.30/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_CODER_NEXT = "qwen/qwen3-coder-next"

// ===========================================================================//
// DeepSeek Models
// ===========================================================================//

// DeepSeek DeepSeek V3.1
// Input $0.20/M Output $0.80/M Context 163,840
const OPENROUTER_MODEL_DEEPSEEK_V3_1 = "deepseek/deepseek-chat-v3.1"

// ===========================================================================//
// xAI Models
// ===========================================================================//

// xAI Grok 3
// Input $3.00/M Output $15.00/M Context 131,072
const OPENROUTER_MODEL_GROK_3 = "x-ai/grok-3"

// xAI Grok 3 Mini
// Input $0.30/M Output $0.50/M Context 131,072
const OPENROUTER_MODEL_GROK_3_MINI = "x-ai/grok-3-mini"

// xAI Grok 4
// Input $3.00/M Output $15.00/M Context 256,000
const OPENROUTER_MODEL_GROK_4 = "x-ai/grok-4"

// ===========================================================================//
// Other Models
// ===========================================================================//

// MoonshotAI Kimi K2.5
// Input $0.45/M Output $2.25/M Context 262,144
const OPENROUTER_MODEL_KIMI_K2_5 = "moonshotai/kimi-k2.5"

// MiniMax M2.1
// Input $0.27/M Output $0.95/M Context 196,608
const OPENROUTER_MODEL_MINIMAX_M2_1 = "minimax/minimax-m2.1"

// ByteDance Seed 1.6
// Input $0.25/M Output $2.00/M Context 262,144
const OPENROUTER_MODEL_SEED_1_6 = "bytedance-seed/seed-1.6"

// ByteDance Seed 1.6 Flash
// Input $0.075/M Output $0.30/M Context 262,144
const OPENROUTER_MODEL_SEED_1_6_FLASH = "bytedance-seed/seed-1.6-flash"

// Xiaomi MiMo-V2-Flash
// Input $0.09/M Output $0.29/M Context 262,144
const OPENROUTER_MODEL_MIMO_V2_FLASH = "xiaomi/mimo-v2-flash"

// Z.AI GLM 4.7
// Input $0.40/M Output $1.50/M Context 202,752
const OPENROUTER_MODEL_GLM_4_7 = "z-ai/glm-4.7"

// Z.AI GLM 4.7 Flash
// Input $0.06/M Output $0.40/M Context 202,752
const OPENROUTER_MODEL_GLM_4_7_FLASH = "z-ai/glm-4.7-flash"

// StepFun Step 3.5 Flash
// Input $0.10/M Output $0.30/M Context 256,000
const OPENROUTER_MODEL_STEP_3_5_FLASH = "stepfun/step-3.5-flash"

// ===========================================================================//
// Image Models
// ===========================================================================//

// Google Gemini 2.5 Flash Image
// Input $0.30/M Output $2.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH_IMAGE = "google/gemini-2.5-flash-image"

// OpenAI GPT-5 Image Mini
// Input $2.50/M Output $2/M Context 1,048,576
const OPENROUTER_MODEL_GPT_5_IMAGE_MINI = "openai/gpt-5-image-mini"

// OpenAI GPT-5 Image
// Input $10.00/M Output $10/M Context 400,000
const OPENROUTER_MODEL_GPT_5_IMAGE = "openai/gpt-5-image"

// ===========================================================================//
// Embedding Models
// ===========================================================================//

// Qwen Qwen3 Embedding 0.6B
// Input $0.01/M Output $0.00/M
const OPENROUTER_MODEL_QWEN_3_EMBEDDING_0_6B = "qwen/qwen3-embedding-0.6b"

// Mistral Mistral Embedding 2312
// Input $0.10/M Output $0.00/M
const OPENROUTER_MODEL_MISTRAL_EMBED_2312 = "mistralai/mistral-embed-2312"

// Google Gemini Embedding 001
// Input $0.15/M Output $0.00/M
const OPENROUTER_MODEL_GEMINI_EMBED_001 = "google/gemini-embedding-001"

// OpenAI Text Embedding Ada 002
// Input $0.10/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_ADA_002 = "openai/text-embedding-ada-002"

// Mistral Codestral Embedding 2505
// Input $0.15/M Output $0.00/M
const OPENROUTER_MODEL_CODESTRAL_EMBED_2505 = "mistralai/codestral-embed-2505"

// OpenAI Text Embedding 3 Large
// Input $0.13/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_3_LARGE = "openai/text-embedding-3-large"

// OpenAI Text Embedding 3 Small
// Input $0.02/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_3_SMALL = "openai/text-embedding-3-small"
