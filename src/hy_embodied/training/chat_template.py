"""SFT chat template.

Training data already embeds ``/think`` / ``/no_think`` suffixes in user
messages and ``<think>...</think><answer>...</answer>`` wrappers in assistant
messages. The inference chat template (bundled with the checkpoint) would
re-apply those markers and produce doubly-nested outputs, so during SFT we use
a data-driven template that passes the data through verbatim.

Only the generation-prompt branch is kept, so the saved template stays
compatible with ``apply_chat_template(..., add_generation_prompt=True)`` at
inference time if needed.
"""

from __future__ import annotations

_SFT_CHAT_TEMPLATE = """\
{%- if not add_generation_prompt is defined %}
    {%- set add_generation_prompt = false %}
{%- endif %}
{%- set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_first_user=true, is_last_user=false) %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if ns.is_first_sp %}
            {%- set ns.system_prompt = ns.system_prompt + message['content'] %}
            {%- set ns.is_first_sp = false %}
        {%- else %}
            {% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{- bos_token }}
{{- ns.system_prompt }}
{%- if ns.system_prompt != '' or tools %}
    {{- '<｜hy_place▁holder▁no▁3｜>' }}
{%- endif %}
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {%- set ns.is_tool = false %}
        {%- set ns.is_first = false %}
        {%- set ns.is_last_user = true %}
        {{- '<｜hy_User｜>'}}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {%- for content in message.content %}
                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}
                    {%- set image_count.value = image_count.value + 1 %}
                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
                    <｜hy_place▁holder▁no▁666｜><｜hy_place▁holder▁no▁669｜><｜hy_place▁holder▁no▁672｜><｜hy_place▁holder▁no▁667｜>
                {%- elif content.type == 'video' or 'video' in content %}
                    {%- set video_count.value = video_count.value + 1 %}
                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}
                    <｜hy_place▁holder▁no▁666｜><｜hy_place▁holder▁no▁670｜><｜hy_place▁holder▁no▁672｜><｜hy_place▁holder▁no▁667｜>
                {%- elif 'text' in content %}
                    {{- content.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- endif %}
    {%- if message['role'] == 'assistant' %}
        {%- set ns.is_last_user = false %}
        {{- '<｜hy_Assistant｜>' }}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- else %}
            {%- for content_item in message['content'] %}
                {%- if 'text' in content_item %}
                    {{- content_item['text'] }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- eos_token }}
    {%- endif %}
    {%- if message['role'] == 'tool' %}
        {%- set ns.is_last_user = false %}
        {%- set ns.is_tool = true %}
        {%- if ns.is_output_first %}
            {{- '<｜hy_User｜>' + '<tool_responses><tool_response>' + message['content'] + '</tool_response>' }}
            {%- set ns.is_output_first = false %}
        {%- else %}
            {{- '\\n<tool_response>' + message['content'] + '</tool_response>' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.is_tool %}
    {{- '</tool_responses>' }}
{%- endif %}
{%- if add_generation_prompt %}
    {{- '<｜hy_Assistant｜>' }}
    {%- if enable_thinking is defined and enable_thinking %}
        {{- '<think>' }}
    {%- else %}
        {{- '<think>\\n\\n</think>\\n' }}
    {%- endif %}
{%- endif %}
"""


def build_sft_chat_template() -> str:
    """Return the training-specific chat template string."""
    return _SFT_CHAT_TEMPLATE


__all__ = ["build_sft_chat_template"]
