# LLM usage costs

Provider-returned `usage.cost` values are preserved, including zero. When that
field is missing or null, calls to `api.aigateway.umn.edu` using either model below
receive an estimated `usage.cost` in USD:

| Gateway model | Input USD / 1M tokens | Output USD / 1M tokens |
| --- | ---: | ---: |
| `gpt-5.6-luna` | 0.20 | 1.20 |
| `gemini-3.8-flash` | 0.75 | 3.75 |

These rates were supplied by the user from the UMN models UI on 2026-09-10.
Recognized provider prefixes, such as `openai/` and `vertex_ai/`, are normalized.
Other gateway hosts and unknown models do not receive these prices.

The estimate is `(input_tokens * input_rate + output_tokens * output_rate) / 1e6`.
Both Chat Completions (`prompt_tokens`, `completion_tokens`) and Responses
(`input_tokens`, `output_tokens`) usage are supported. Both counts must be present
and nonnegative integers. Cached input is charged at the supplied input rate;
reasoning tokens are already included in output tokens and are not added again.
Cache discounts and other billing adjustments may make actual charges differ.

Calculated usage includes `cost_source: "umn_model_pricing"`,
`cost_is_estimate: true`, `cost_currency: "USD"`, and `cost_pricing` with the model
and rates used. For example, Luna's 12 input and 47 output tokens cost an estimated
`0.0000588` USD.

Generation responses (including tool rounds) store this under
`llm_reasoning.calls[].responses[].usage`. Confidence responses also retain the
enriched usage. Existing checkpoint and evolution trace serialization preserves
these fields automatically. The change applies to newly captured responses;
previously saved runs are not rewritten. This calculation does not fetch HTTP
cost headers or add run-level cost totals.
