# Picard Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `%picard [topic]` command that shares random Captain Picard facts using the ask infrastructure with a Picard-themed system prompt.

**Architecture:** Thin wrapper over the ask pipeline. A `system_prompt` override parameter is added to `LLMService.completion()` so picard can inject its own personality prompt while reusing ask's model, API key, and rate limits. Conversation context is shared with ask.

**Tech Stack:** Python, Limnoria, LiteLLM (existing stack)

---

### Task 1: Add `system_prompt` override to `completion()`

**Files:**
- Modify: `plugins/llm/src/llm/service.py:1469-1543` (completion method)
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Write the failing test**

In `plugins/llm/tests/test_service.py`, find the `TestCompletion` class and add:

```python
def test_completion_uses_system_prompt_override_when_provided(self) -> None:
    """GIVEN system_prompt kwarg WHEN completion called THEN uses it instead of registry lookup."""
    service, plugin = self.service, self.plugin
    mock_response = self._make_response("Picard response")
    with mock.patch("llm.service.litellm.completion", return_value=mock_response):
        result = service.completion(
            "test prompt",
            command="ask",
            system_prompt="You are Captain Picard.",
            irc=self.mock_irc,
            msg=self.mock_msg,
        )

    assert result.content == "Picard response"
    # Verify the system prompt override was used, not the registry value
    call_args = litellm.completion.call_args
    messages = call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert any("Captain Picard" in m["content"] for m in system_msgs)
    assert not any("You are helpful" in m["content"] for m in system_msgs)
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `completion()` doesn't accept `system_prompt` parameter yet.

**Step 3: Write minimal implementation**

In `plugins/llm/src/llm/service.py`, modify the `completion` method signature (around line 1469):

Change:
```python
def completion(
    self,
    prompt: str,
    command: str = "ask",
    images: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    channel_history: list[dict[str, str]] | None = None,
    irc: Irc | None = None,
    msg: IrcMsg | None = None,
) -> CompletionResult:
```

To:
```python
def completion(
    self,
    prompt: str,
    command: str = "ask",
    images: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    channel_history: list[dict[str, str]] | None = None,
    irc: Irc | None = None,
    msg: IrcMsg | None = None,
    system_prompt: str | None = None,
) -> CompletionResult:
```

Update the docstring Args section to add:
```
            system_prompt: Optional system prompt override. When provided, used
                instead of the ``{command}SystemPrompt`` registry value.
```

Then modify lines 1539-1543 from:
```python
            model = self.plugin.registryValue(f"{command}Model", channel)
            base_system_prompt = self.plugin.registryValue(f"{command}SystemPrompt", channel)

            # Build system prompt (context now injected as user message in _build_messages)
            system_prompt = self._build_system_prompt(base_system_prompt)
```

To:
```python
            model = self.plugin.registryValue(f"{command}Model", channel)
            if system_prompt is None:
                base_system_prompt = self.plugin.registryValue(
                    f"{command}SystemPrompt", channel
                )
            else:
                base_system_prompt = system_prompt

            # Build system prompt (context now injected as user message in _build_messages)
            built_system_prompt = self._build_system_prompt(base_system_prompt)
```

And update the `_build_messages` call to use `built_system_prompt`:
```python
            messages = self._build_messages(
                prompt, images, history, channel_history, built_system_prompt, irc, msg
            )
```

**Step 4: Run test to verify it passes**

Run: `make test`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: add system_prompt override to completion()"
```

---

### Task 2: Add `picardSystemPrompt` config

**Files:**
- Modify: `plugins/llm/src/llm/config.py:145-166` (after askSystemPrompt section)
- Modify: `plugins/llm/tests/conftest.py:116` (add to registry defaults)

**Step 1: Add the config value**

In `plugins/llm/src/llm/config.py`, after the `askSystemPrompt` registration (around line 155), add:

```python
conf.registerChannelValue(
    LLM,
    "picardSystemPrompt",
    registry.String(
        "You are Captain Jean-Luc Picard of the USS Enterprise. "
        "Share an interesting, surprising, or amusing fact — it can be about you, "
        "Starfleet, the Enterprise crew, or the Star Trek universe. "
        "Draw inspiration from the ongoing conversation when relevant. "
        "Stay in character. Be concise (1-3 sentences for IRC). "
        "If given a topic, relate your fact to it.",
        _("""System prompt for picard command — defines Picard personality"""),
    ),
)
```

**Step 2: Add to test fixture defaults**

In `plugins/llm/tests/conftest.py`, add to the `defaults` dict in `make_registry_side_effect()` after the ask block (around line 116):

```python
        # Picard command (reuses ask model/key)
        "picardSystemPrompt": "You are Captain Picard.",
```

**Step 3: Run tests to verify nothing breaks**

Run: `make test`
Expected: PASS (no behavioral change yet)

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat: add picardSystemPrompt config value"
```

---

### Task 3: Add `picard` command to plugin

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (after the `ask` method, around line 1540)
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Write the failing test**

In `plugins/llm/tests/test_commands.py`, add a new test class after `TestAskCommand`:

```python
class TestPicardCommand:
    """Tests for the real LLM.picard method."""

    def test_picard_replies_with_completion_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN no topic WHEN picard called THEN replies with Picard fact."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Tea, Earl Grey, hot. A fine choice.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.picard(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once_with(
            "Tea, Earl Grey, hot. A fine choice.", prefixNick=False
        )

    def test_picard_with_topic(self, plugin_env, mocker: MockerFixture):
        """GIVEN a topic WHEN picard called THEN passes topic as prompt."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="The Borg are relentless.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.picard(mock_irc, mock_msg, ["the Borg"])

        # Verify completion was called with the topic as prompt
        plugin.llm_service.completion.assert_called_once()
        call_kwargs = plugin.llm_service.completion.call_args
        assert "Borg" in call_kwargs.kwargs.get("prompt", call_kwargs.args[0] if call_kwargs.args else "")

    def test_picard_passes_system_prompt_override(self, plugin_env, mocker: MockerFixture):
        """GIVEN picard command WHEN completion called THEN system_prompt kwarg is set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Make it so.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.picard(mock_irc, mock_msg, [])

        call_kwargs = plugin.llm_service.completion.call_args.kwargs
        assert call_kwargs["command"] == "ask"
        assert "system_prompt" in call_kwargs
        assert "Picard" in call_kwargs["system_prompt"]

    def test_picard_stores_context(self, plugin_env, mocker: MockerFixture):
        """GIVEN successful picard response WHEN called THEN context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Engage!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.picard(mock_irc, mock_msg, [])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[1]["content"] == "Engage!"

    def test_picard_logs_usage_as_picard(self, plugin_env, mocker: MockerFixture):
        """GIVEN picard command WHEN usage logged THEN command is 'picard'."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Indeed.",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.005,
            model="gpt-4",
        )

        plugin.picard(mock_irc, mock_msg, ["diplomacy"])

        plugin.db.log_usage.assert_called_once()
        call_kwargs = plugin.db.log_usage.call_args
        assert call_kwargs.args[2] == "picard"  # command arg
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `plugin.picard` doesn't exist yet.

**Step 3: Write the picard command**

In `plugins/llm/src/llm/plugin.py`, after the `ask = wrap(ask, ...)` line (line 1540), add:

```python
    def picard(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str = "",
    ) -> None:
        """[<topic>]

        Share a random Captain Picard fact. Optionally provide a topic
        to steer the fact (e.g., %picard tea, %picard diplomacy).

        Examples:
          %picard
          %picard the Borg
          %picard Earl Grey tea
        """
        if self._is_old_message(msg):
            return

        pf = self._run_preflight(
            irc,
            msg,
            text or "random fact",
            "ask",
            require_account=False,
        )
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        with self._trace_request("picard", nick, channel):
            prompt = text if text else "Tell me a random Picard fact."

            if self._get_context_enabled(channel):
                ctx_cfg = self._get_context_config(channel)
                history = self.context.get_messages(nick, channel, config=ctx_cfg)
                channel_history = self.context.get_channel_messages(
                    channel, exclude_nick=nick, config=ctx_cfg
                )
            else:
                history, channel_history = [], []

            picard_prompt = self.registryValue("picardSystemPrompt", channel)

            with self._allow_concurrent():
                result = self.llm_service.completion(
                    prompt,
                    command="ask",
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    system_prompt=picard_prompt,
                )

                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                is_action = response.startswith("/me ") and len(response) > 4
                if is_action:
                    action_text = response[4:]
                    if result.grounding_used:
                        action_text = f"{GROUNDING_ICON} {action_text}"
                    target = msg.args[0]
                    irc.queueMsg(ircmsgs.action(target, action_text))
                    response = f"* {irc.nick} {action_text}"
                else:
                    display_response = (
                        f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                    )
                    irc.reply(display_response, prefixNick=False)

            self._store_context_and_log_usage(
                nick, channel, "picard", text or prompt, response, result, irc, msg
            )

    picard = wrap(picard, [("checkCapability", "llm.ask"), optional("text")])
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS

**Step 5: Run full preflight**

Run: `make preflight`
Expected: PASS (format + lint + typecheck + test)

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat: add %picard command for random Picard facts"
```
