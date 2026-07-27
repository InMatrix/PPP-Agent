import httpx

from ppp_simplified.providers import OpenAICompatibleAgent, parse_json_object


def test_parse_qwen_thinking_then_json() -> None:
    payload = parse_json_object(
        '<think>Inspect first.</think>\n'
        '{"tool":"list_files","arguments":{"path":""}}'
    )
    assert payload["tool"] == "list_files"


def test_parse_fenced_json() -> None:
    payload = parse_json_object(
        '```json\n{"tool":"finish","arguments":{"functions":[]}}\n```'
    )
    assert payload["tool"] == "finish"


def test_openai_compatible_qwen_adapter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "Qwen/Qwen3.5-4B"
        assert "Return exactly one JSON object" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<think>Need a file list.</think>'
                                '{"tool":"list_files",'
                                '"arguments":{"path":""},'
                                '"reasoning":"inspect"}'
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    action = OpenAICompatibleAgent(client=client).next_action(
        system_prompt="system",
        messages=[{"role": "user", "content": "issue"}],
    )
    assert action.tool == "list_files"


def test_openai_compatible_qwen_uses_reasoning_content_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": (
                                '{"tool":"search_code",'
                                '"arguments":{"query":"Blueprint"},'
                                '"reasoning":"locate the symbol"}'
                            ),
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    action = OpenAICompatibleAgent(client=client).next_action(
        system_prompt="system",
        messages=[{"role": "user", "content": "issue"}],
    )
    assert action.tool == "search_code"


def test_openai_compatible_qwen_restricts_finalization_to_finish() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        tool_schema = payload["response_format"]["json_schema"]["schema"][
            "properties"
        ]["tool"]
        assert tool_schema["enum"] == ["finish"]
        assert "Only these tools are permitted now: finish." in payload[
            "messages"
        ][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"tool":"finish",'
                                '"arguments":{"functions":["pkg/mod.py:run"]},'
                                '"reasoning":"best supported answer"}'
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    action = OpenAICompatibleAgent(client=client).next_action(
        system_prompt="final turn",
        messages=[{"role": "user", "content": "issue"}],
        allowed_tools=("finish",),
    )
    assert action.tool == "finish"
