package llm

// ===========================================================================//
// OpenAI Models
// ===========================================================================//

// OpenAI GPT-OSS-20B
// Released: 2025-08
// Input $0.04/M Output $0.15/M Context 131,072
const OPENROUTER_MODEL_GPT_OSS_20B = "openai/gpt-oss-20b"

// OpenAI GPT-OSS-120B
// Released: 2025-08
// Input $0.072/M Output $0.28/M Context 131,072
const OPENROUTER_MODEL_GPT_OSS_120B = "openai/gpt-oss-120b"

// OpenAI O4 Mini
// Released: 2025
// Input $1.10/M Output $4.40/M Context 200,000
const OPENROUTER_MODEL_O4_MINI = "openai/o4-mini"

// OpenAI GPT-4.1 Nano
// Released: 2025
// Input $0.10/M Output $0.40/M Context 1,047,576
const OPENROUTER_MODEL_GPT_4_1_NANO = "openai/gpt-4.1-nano"

// OpenAI GPT-5 Nano
// Released: 2025
// Input $0.05/M Output $0.40/M Context 400,000
const OPENROUTER_MODEL_GPT_5_NANO = "openai/gpt-5-nano"

// OpenAI GPT-5.1
// Released: 2025
// Input $1.25/M Output $10.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_1 = "openai/gpt-5.1"

// OpenAI GPT-5.2
// Released: 2026
// Input $1.75/M Output $14.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2 = "openai/gpt-5.2"

// OpenAI GPT-5.2 Chat (Instant)
// Released: 2026
// Input $1.75/M Output $14.00/M Context 128,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_CHAT = "openai/gpt-5.2-chat"

// OpenAI GPT-5.2 Pro
// Released: 2026
// Input $21.00/M Output $168.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_PRO = "openai/gpt-5.2-pro"

// OpenAI GPT-5.2 Codex
// Released: 2026
// Input $1.75/M Output $14.00/M Context 400,000 Web Search: $10/K
const OPENROUTER_MODEL_GPT_5_2_CODEX = "openai/gpt-5.2-codex"

// OpenAI GPT-6 Astra
// Released: 2026-09
// Input $10/M Output $50/M Context 1,050,000
const OPENROUTER_MODEL_GPT_6_ASTRA = "openai/gpt-6-astra"

// OpenAI GPT Astra Latest (alias — always redirects to the latest Astra)
// Released: 2026-09
// Input $10/M Output $50/M Context 1,050,000
const OPENROUTER_MODEL_GPT_ASTRA_LATEST = "openai/gpt-astra-latest"

// OpenAI GPT Sol Latest (alias — always redirects to the latest Sol)
// Released: 2026-09
// Input $2/M Output $10/M Context 1,050,000
const OPENROUTER_MODEL_GPT_SOL_LATEST = "openai/gpt-sol-latest"

// OpenAI GPT Terra Latest (alias — always redirects to the latest Terra)
// Released: 2026-09
// Input $2/M Output $12/M Context 1,050,000
const OPENROUTER_MODEL_GPT_TERRA_LATEST = "openai/gpt-terra-latest"

// OpenAI GPT Luna Latest (alias — always redirects to the latest Luna)
// Released: 2026-09
// Input $0.20/M Output $1.20/M Context 1,050,000
const OPENROUTER_MODEL_GPT_LUNA_LATEST = "openai/gpt-luna-latest"

// ===========================================================================//
// Anthropic Models
// ===========================================================================//

// Anthropic Claude Sonnet 4
// Released: 2025
// Input $3.00/M Output $15.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"

// Anthropic Claude Sonnet 4.5
// Released: 2025
// Input $3.00/M Output $15.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"

// Anthropic Claude Haiku 4.5
// Released: 2025
// Input $0.80/M Output $4.00/M Context 200,000
const OPENROUTER_MODEL_CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"

// Anthropic Claude Opus 4.5
// Released: 2025
// Input $5.00/M Output $25.00/M Context 200,000
const OPENROUTER_MODEL_CLAUDE_OPUS_4_5 = "anthropic/claude-opus-4.5"

// Anthropic Claude Opus 4.6
// Released: 2025
// Input $5.00/M Output $25.00/M Context 1,000,000
const OPENROUTER_MODEL_CLAUDE_OPUS_4_6 = "anthropic/claude-opus-4.6"

// ===========================================================================//
// Google Models
// ===========================================================================//

// Google Gemma 3 12B
// Released: 2025
// Input $0.048/M Output $0.193/M Context 96,000
const OPENROUTER_MODEL_GEMMA_3_12B_IT = "google/gemma-3-12b-it"

// Google Gemma 3 27B
// Released: 2025
// Input $0.067/M Output $0.267/M Context 96,000
const OPENROUTER_MODEL_GEMMA_3_27B_IT = "google/gemma-3-27b-it"

// Google Gemma 4 26B A4B IT
// Released: 2026
// Input $0.07/M Output $0.34/M Context 262,144
const OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT = "google/gemma-4-26b-a4b-it"

// Google Gemma 4 31B IT
// Released: 2026
// Input $0.09/M Output $0.34/M Context 262,144
const OPENROUTER_MODEL_GEMMA_4_31B_IT = "google/gemma-4-31b-it"

// Google Gemini 2.5 Flash Lite
// Released: 2025
// Input $0.10/M Output $0.40/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"

// Google Gemini 2.5 Flash
// Released: 2025
// Input $0.30/M Output $2.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH = "google/gemini-2.5-flash"

// Google Gemini 2.5 Pro
// Released: 2025
// Input $1.25/M Output $10/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_PRO = "google/gemini-2.5-pro"

// Google Gemini 3.1 Flash Lite
// Released: 2026
// Input $0.25/M Output $1.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_1_FLASH_LITE = "google/gemini-3.1-flash-lite"

// Google Gemini 3 Flash Preview
// Released: 2026
// Input $0.50/M Output $3.00/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"

// Google Gemini 3 Pro Preview
// Released: 2026
// Input $2/M Output $12/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"

// Google Gemini 3.5 Flash
// Released: 2026-05
// Input $1.50/M Output $9.00/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_5_FLASH = "google/gemini-3.5-flash"

// Google Gemini 3.5 Flash Lite
// Released: 2026-07
// Input $0.30/M Output $2.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_5_FLASH_LITE = "google/gemini-3.5-flash-lite"

// Google Gemini 3.6 Flash
// Released: 2026
// Input $0.75/M Output $3.75/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_6_FLASH = "google/gemini-3.6-flash"

// Google Gemini 3.7 Flash
// Released: 2026
// Input $0.75/M Output $3.75/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_3_7_FLASH = "google/gemini-3.7-flash"

// Google Gemini 3.8 Flash
// Released: 2026
// Input $0.75/M Output $3.75/M Context 1,048,576 (introductory price)
const OPENROUTER_MODEL_GEMINI_3_8_FLASH = "google/gemini-3.8-flash"

// ===========================================================================//
// Mistral Models
// ===========================================================================//

// Mistral Mistral Nemo
// Released: 2024
// Input $0.02/M Output $0.04/M Context 131,072
const OPENROUTER_MODEL_MISTRAL_NEMO = "mistralai/mistral-nemo"

// Mistral Mistral Small 3.2 24B Instruct
// Released: 2025
// Input $0.075/M Output $0.20/M Context 131,072
const OPENROUTER_MODEL_MISTRAL_SMALL_3_2_24B_INSTRUCT = "mistralai/mistral-small-3.2-24b-instruct"

// Mistral Mistral Medium 3.1
// Released: 2025
// Input $0.40/M Output $2/M Context 131,072
const OPENROUTER_MODEL_MISTRAL_MEDIUM_3_1 = "mistralai/mistral-medium-3.1"

// Mistral Devstral 2512
// Released: 2025-12
// Input $0.05/M Output $0.22/M Context 262,144
const OPENROUTER_MODEL_DEVSTRAL_2512 = "mistralai/devstral-2512"

// ===========================================================================//
// Qwen Models
// ===========================================================================//

// Qwen Qwen3 235B A22B Instruct 2507
// Released: 2025-07
// Input $0.078/M Output $0.312/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_235B_A22B_INSTRUCT_2507 = "qwen/qwen3-235b-a22b-2507"

// Qwen Qwen3 30B A3B
// Released: 2025
// Input $0.02/M Output $0.08/M Context 40,960
const OPENROUTER_MODEL_QWEN_3_30B_A3B = "qwen/qwen3-30b-a3b"

// Qwen Qwen3 Max Thinking
// Released: 2025
// Input $1.20/M Output $6.00/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_MAX_THINKING = "qwen/qwen3-max-thinking"

// Qwen Qwen3 Coder Next
// Released: 2025
// Input $0.07/M Output $0.30/M Context 262,144
const OPENROUTER_MODEL_QWEN_3_CODER_NEXT = "qwen/qwen3-coder-next"

// Qwen Qwen3.7 Flash
// Released: 2026
// Input $0.03/M Output $0.13/M Context 1,048,576
const OPENROUTER_MODEL_QWEN_3_7_FLASH = "qwen/qwen3.7-flash"

// Qwen Qwen3.8 Flash
// Released: 2026-08
// Input $0.15/M Output $0.47/M Context 1,000,000
const OPENROUTER_MODEL_QWEN_3_8_FLASH = "qwen/qwen3.8-flash"

// Qwen Qwen3.8 Max
// Released: 2026
// Input $2/M Output $6/M Context 1,000,000
const OPENROUTER_MODEL_QWEN_3_8_MAX = "qwen/qwen3.8-max"

// ===========================================================================//
// Meta Llama Models
// ===========================================================================//

// Meta Llama 3.2 1B Instruct
// Released: 2024
// Input $0.027/M Output $0.20/M Context 60,000
const OPENROUTER_MODEL_LLAMA_3_2_1B_INSTRUCT = "meta-llama/llama-3.2-1b-instruct"

// Meta Llama 3.1 8B Instruct
// Released: 2024
// Input $0.05/M Output $0.08/M Context 131,072
const OPENROUTER_MODEL_LLAMA_3_1_8B_INSTRUCT = "meta-llama/llama-3.1-8b-instruct"

// Meta Llama 3.3 70B Instruct
// Released: 2024
// Input $0.10/M Output $0.32/M Context 131,072
const OPENROUTER_MODEL_LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"

// Meta Llama 4 Scout
// Released: 2025
// Input $0.10/M Output $0.30/M Context 1,000,000
const OPENROUTER_MODEL_LLAMA_4_SCOUT = "meta-llama/llama-4-scout"

// ===========================================================================//
// IBM Granite Models
// ===========================================================================//

// IBM Granite 4.0 H Micro
// Released: 2025
// Input $0.017/M Output $0.11/M Context 131,072
const OPENROUTER_MODEL_GRANITE_4_0_H_MICRO = "ibm-granite/granite-4.0-h-micro"

// ===========================================================================//
// DeepSeek Models
// ===========================================================================//

// DeepSeek DeepSeek V3.1
// Released: 2025
// Input $0.20/M Output $0.80/M Context 163,840
const OPENROUTER_MODEL_DEEPSEEK_V3_1 = "deepseek/deepseek-chat-v3.1"

// DeepSeek DeepSeek V4.1 Flash
// Released: 2026-09
// Input $0.15/M Output $0.60/M Context 1,050,000
const OPENROUTER_MODEL_DEEPSEEK_V4_1_FLASH = "deepseek/deepseek-v4.1-flash"

// DeepSeek DeepSeek V4 Flash 0731
// Released: 2026-07
// Input $0.05/M Output $0.16/M Context 1,310,720
const OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_0731 = "deepseek/deepseek-v4-flash-0731"

// ===========================================================================//
// xAI Models
// ===========================================================================//

// xAI Grok 3
// Released: 2025
// Input $3.00/M Output $15.00/M Context 131,072
const OPENROUTER_MODEL_GROK_3 = "x-ai/grok-3"

// xAI Grok 3 Mini
// Released: 2025
// Input $0.30/M Output $0.50/M Context 131,072
const OPENROUTER_MODEL_GROK_3_MINI = "x-ai/grok-3-mini"

// xAI Grok 4
// Released: 2025
// Input $3.00/M Output $15.00/M Context 256,000
const OPENROUTER_MODEL_GROK_4 = "x-ai/grok-4"

// ===========================================================================//
// Sakana Models
// ===========================================================================//

// Sakana Fugu Ultra v2
// Released: 2026-09
// Input $5/M Output $30/M Context 1,000,000
const OPENROUTER_MODEL_FUGU_ULTRA_V2 = "sakana/fugu-ultra-v2"

// Sakana Fugu Max
// Released: 2026-09
// Input $2/M Output $6/M Context 1,000,000
const OPENROUTER_MODEL_FUGU_MAX = "sakana/fugu-max"

// ===========================================================================//
// InclusionAI Models
// ===========================================================================//

// InclusionAI Ling 3.0 Flash
// Released: 2026-07
// Input $0.021/M Output $0.063/M Context 262,144
const OPENROUTER_MODEL_LING_3_0_FLASH = "inclusionai/ling-3.0-flash"

// InclusionAI Ling 3.0 Flash Fin (finance-flavored variant)
// Released: 2026
// Input $0.06/M Output $0.18/M Context 262,144
const OPENROUTER_MODEL_LING_3_0_FLASH_FIN = "inclusionai/ling-3.0-flash-fin"

// InclusionAI Ling 3.0 Flash VL
// Released: 2026-09
// Input $0.06/M Output $0.18/M Context 131,072
const OPENROUTER_MODEL_LING_3_0_FLASH_VL = "inclusionai/ling-3.0-flash-vl"

// ===========================================================================//
// Inception Models
// ===========================================================================//

// Inception Mercury 2.5
// Released: 2026-09
// Input $0.04/M Output $0.15/M Context 260,000
const OPENROUTER_MODEL_MERCURY_2_5 = "inception/mercury-2.5"

// ===========================================================================//
// Inference.net Models
// ===========================================================================//

// Inference.net Schematron V2 Turbo
// Released: 2026-09
// Input $0.03/M Output $0.15/M Context 128,000
const OPENROUTER_MODEL_SCHEMATRON_V2_TURBO = "inference-net/schematron-v2-turbo"

// Inference.net Schematron V2 Small
// Released: 2026-09
// Input $0.05/M Output $0.23/M Context 128,000
const OPENROUTER_MODEL_SCHEMATRON_V2_SMALL = "inference-net/schematron-v2-small"

// ===========================================================================//
// Other Models
// ===========================================================================//

// MoonshotAI Kimi K2.5
// Released: 2025
// Input $0.45/M Output $2.25/M Context 262,144
const OPENROUTER_MODEL_KIMI_K2_5 = "moonshotai/kimi-k2.5"

// MiniMax M2.1
// Released: 2025
// Input $0.27/M Output $0.95/M Context 196,608
const OPENROUTER_MODEL_MINIMAX_M2_1 = "minimax/minimax-m2.1"

// MiniMax M3
// Released: 2026
// Input $0.23/M Output $0.96/M Context 1,000,000
const OPENROUTER_MODEL_MINIMAX_M3 = "minimax/minimax-m3"

// ByteDance Seed 1.6
// Released: 2025
// Input $0.25/M Output $2.00/M Context 262,144
const OPENROUTER_MODEL_SEED_1_6 = "bytedance-seed/seed-1.6"

// ByteDance Seed 1.6 Flash
// Released: 2025
// Input $0.075/M Output $0.30/M Context 262,144
const OPENROUTER_MODEL_SEED_1_6_FLASH = "bytedance-seed/seed-1.6-flash"

// Xiaomi MiMo-V2-Flash
// Released: 2025
// Input $0.09/M Output $0.29/M Context 262,144
const OPENROUTER_MODEL_MIMO_V2_FLASH = "xiaomi/mimo-v2-flash"

// Z.AI GLM 4.7
// Released: 2025
// Input $0.40/M Output $1.50/M Context 202,752
const OPENROUTER_MODEL_GLM_4_7 = "z-ai/glm-4.7"

// Z.AI GLM 4.7 Flash
// Released: 2025
// Input $0.06/M Output $0.40/M Context 202,752
const OPENROUTER_MODEL_GLM_4_7_FLASH = "z-ai/glm-4.7-flash"

// Z.AI GLM 5.3 Flash
// Released: 2026
// Input $0.075/M Output $0.25/M Context 1,310,720
const OPENROUTER_MODEL_GLM_5_3_FLASH = "z-ai/glm-5.3-flash"

// StepFun Step 3.5 Flash
// Released: 2025
// Input $0.10/M Output $0.30/M Context 256,000
const OPENROUTER_MODEL_STEP_3_5_FLASH = "stepfun/step-3.5-flash"

// ===========================================================================//
// Image Models
// ===========================================================================//

// Google Gemini 2.5 Flash Image
// Released: 2025
// Input $0.30/M Output $2.50/M Context 1,048,576
const OPENROUTER_MODEL_GEMINI_2_5_FLASH_IMAGE = "google/gemini-2.5-flash-image"

// OpenAI GPT-5 Image Mini
// Released: 2025
// Input $2.50/M Output $2/M Context 1,048,576
const OPENROUTER_MODEL_GPT_5_IMAGE_MINI = "openai/gpt-5-image-mini"

// OpenAI GPT-5 Image
// Released: 2025
// Input $10.00/M Output $10/M Context 400,000
const OPENROUTER_MODEL_GPT_5_IMAGE = "openai/gpt-5-image"

// ===========================================================================//
// Embedding Models
// ===========================================================================//

// Qwen Qwen3 Embedding 0.6B
// Released: 2025
// Input $0.01/M Output $0.00/M
const OPENROUTER_MODEL_QWEN_3_EMBEDDING_0_6B = "qwen/qwen3-embedding-0.6b"

// Mistral Mistral Embedding 2312
// Released: 2023-12
// Input $0.10/M Output $0.00/M
const OPENROUTER_MODEL_MISTRAL_EMBED_2312 = "mistralai/mistral-embed-2312"

// Google Gemini Embedding 001
// Released: 2024
// Input $0.15/M Output $0.00/M
const OPENROUTER_MODEL_GEMINI_EMBED_001 = "google/gemini-embedding-001"

// OpenAI Text Embedding Ada 002
// Released: 2022
// Input $0.10/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_ADA_002 = "openai/text-embedding-ada-002"

// Mistral Codestral Embedding 2505
// Released: 2025-05
// Input $0.15/M Output $0.00/M
const OPENROUTER_MODEL_CODESTRAL_EMBED_2505 = "mistralai/codestral-embed-2505"

// OpenAI Text Embedding 3 Large
// Released: 2024
// Input $0.13/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_3_LARGE = "openai/text-embedding-3-large"

// OpenAI Text Embedding 3 Small
// Released: 2024
// Input $0.02/M Output $0.00/M
const OPENROUTER_MODEL_TEXT_EMBEDDING_3_SMALL = "openai/text-embedding-3-small"
