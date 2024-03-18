TOKEN_COSTS = {}

# Priced in USD per token
TOKEN_COSTS["You.com"] = {}
TOKEN_COSTS["You.com"]["completion"] = 26 / 10_000_000
TOKEN_COSTS["You.com"]["prompt"] = 19.5 / 10_000_000
TOKEN_COSTS["You.com"]["request"] = 0.0049

TOKEN_COSTS["youcom/rag"] = TOKEN_COSTS["You.com"]

TOKEN_COSTS["Perplexity pplx_70b_online"] = {}
TOKEN_COSTS["Perplexity pplx_70b_online"]["completion"] = 28 / 10_000_000
TOKEN_COSTS["Perplexity pplx_70b_online"]["prompt"] = 7 / 10_000_000
TOKEN_COSTS["Perplexity pplx_70b_online"]["request"] = 0.005

TOKEN_COSTS["Perplexity pplx_7b_online"] = {}
TOKEN_COSTS["Perplexity pplx_7b_online"]["completion"] = 2.8 / 10_000_000
TOKEN_COSTS["Perplexity pplx_7b_online"]["prompt"] = 0.7 / 10_000_000
TOKEN_COSTS["Perplexity pplx_7b_online"]["request"] = 0.005

TOKEN_COSTS["perplexity/sonar-medium-online"] = TOKEN_COSTS[
    "Perplexity pplx_70b_online"
]
TOKEN_COSTS["perplexity/sonar-small-online"] = TOKEN_COSTS["Perplexity pplx_7b_online"]

TOKEN_COSTS["mistralai/mixtral-8x7b-chat"] = {}
TOKEN_COSTS["mistralai/mixtral-8x7b-chat"]["completion"] = 0.6 / 10_000_000
TOKEN_COSTS["mistralai/mixtral-8x7b-chat"]["prompt"] = 0.6 / 10_000_000

TOKEN_COSTS["mistralai/Mixtral-8x7B-Instruct-v0.1"] = TOKEN_COSTS[
    "mistralai/mixtral-8x7b-chat"
]

TOKEN_COSTS["mistralai/mistral-7b-chat"] = {}
TOKEN_COSTS["mistralai/mistral-7b-chat"]["completion"] = 0.2 / 10_000_000
TOKEN_COSTS["mistralai/mistral-7b-chat"]["prompt"] = 0.2 / 10_000_000

TOKEN_COSTS["mistralai/Mistral-7B-Instruct-v0.1"] = TOKEN_COSTS[
    "mistralai/mistral-7b-chat"
]

TOKEN_COSTS["meta/code-llama-instruct-34b-chat"] = {}
TOKEN_COSTS["meta/code-llama-instruct-34b-chat"]["completion"] = 0.776 / 10_000_000
TOKEN_COSTS["meta/code-llama-instruct-34b-chat"]["prompt"] = 0.776 / 10_000_000

TOKEN_COSTS["togethercomputer/CodeLlama-34b-Instruct"] = TOKEN_COSTS[
    "meta/code-llama-instruct-34b-chat"
]

TOKEN_COSTS["WizardLM/WizardLM-13B-V1.2"] = {}
TOKEN_COSTS["WizardLM/WizardLM-13B-V1.2"]["completion"] = 0.3 / 10_000_000
TOKEN_COSTS["WizardLM/WizardLM-13B-V1.2"]["prompt"] = 0.3 / 10_000_000

TOKEN_COSTS["zero-one-ai/Yi-34B-Chat"] = {}
TOKEN_COSTS["zero-one-ai/Yi-34B-Chat"]["completion"] = 0.8 / 10_000_000
TOKEN_COSTS["zero-one-ai/Yi-34B-Chat"]["prompt"] = 0.8 / 10_000_000

TOKEN_COSTS["gpt-3.5-turbo-1106"] = {}
TOKEN_COSTS["gpt-3.5-turbo-1106"]["completion"] = 2.0 / 10_000_000
TOKEN_COSTS["gpt-3.5-turbo-1106"]["prompt"] = 1.0 / 10_000_000

TOKEN_COSTS["claude-instant-v1"] = {}
TOKEN_COSTS["claude-instant-v1"]["completion"] = 2.40 / 10_000_000
TOKEN_COSTS["claude-instant-v1"]["prompt"] = 0.80 / 10_000_000

TOKEN_COSTS["claude-v1"] = {}
TOKEN_COSTS["claude-v1"]["completion"] = 24.0 / 10_000_000
TOKEN_COSTS["claude-v1"]["prompt"] = 8.0 / 10_000_000

TOKEN_COSTS["claude-1.3"] = TOKEN_COSTS["claude-v1"]

TOKEN_COSTS["claude-v2"] = {}
TOKEN_COSTS["claude-v2"]["completion"] = 24.0 / 10_000_000
TOKEN_COSTS["claude-v2"]["prompt"] = 8.0 / 10_000_000

TOKEN_COSTS["gpt-4-1106-preview"] = {}
TOKEN_COSTS["gpt-4-1106-preview"]["completion"] = 30.0 / 10_000_000
TOKEN_COSTS["gpt-4-1106-preview"]["prompt"] = 10.0 / 10_000_000

TOKEN_COSTS["meta/llama-2-70b-chat"] = {}
TOKEN_COSTS["meta/llama-2-70b-chat"]["completion"] = 0.9 / 10_000_000
TOKEN_COSTS["meta/llama-2-70b-chat"]["prompt"] = 0.9 / 10_000_000

TOKEN_COSTS["togethercomputer/llama-2-70b-chat"] = TOKEN_COSTS["meta/llama-2-70b-chat"]

TOKEN_COSTS["no_model_correct"] = {}
TOKEN_COSTS["no_model_correct"]["completion"] = 0.0
TOKEN_COSTS["no_model_correct"]["prompt"] = 0.0


MODELS_TO_ROUTE_NAMES = {
    "gpt-3.5-turbo-1106": "GPT-3.5",
    "claude-instant-v1": "Claude Instant V1",
    "claude-v1": "Claude V1",
    "claude-v2": "Claude V2",
    "gpt-4-1106-preview": "GPT-4",
    "meta/llama-2-70b-chat": "Llama 70B",
    "mistralai/mixtral-8x7b-chat": "Mixtral 8x7B",
    "zero-one-ai/Yi-34B-Chat": "Yi 34B",
    "WizardLM/WizardLM-13B-V1.2": "WizardLM 13B",
    "meta/code-llama-instruct-34b-chat": "Code Llama 34B",
    "mistralai/mistral-7b-chat": "Mistral 7B",
    "oracle": "Oracle",
    "you.com": "You.com",
    "pplx_70b": "PPLX 70B",
    "pplx_7b": "PPLX 7B",
    "openai/gpt-3.5-turbo": "GPT-3.5",
    "openai/openai/gpt-4-1106-preview": "GPT-4",
    "youcom/rag": "You.com",
    "perplexity/sonar-medium-online": "Sonar Medium",
    "perplexity/sonar-small-online": "Sonar Small",
}
