# Emote/Action Response Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the bot to respond with IRC actions (`/me`) when the LLM decides an emote feels more natural than a direct reply.

**Architecture:** Relax command prefix sanitization to allow `/` through, detect `/me ` prefix in `ask` responses, and send via `ircmsgs.action()` instead of `irc.reply()`. Add a one-sentence nudge in the code-level system prompt builder so the model knows it can use `/me`.

**Tech Stack:** Python, Limnoria (`ircmsgs.action`), pytest

---

### Task 1: Update default command prefixes

**Files:**
- Modify: `plugins/llm/src/llm/config.py:416-417`

**Step 1: Change the default**

In `config.py`, change the `commandPrefixes` default from `[".", "/"]` to `["."]` and update the help text:

```python
conf.registerGlobalValue(
    LLM,
    "commandPrefixes",
    registry.SpaceSeparatedListOfStrings(
        ["."],
        _("""Command prefixes to sanitize in output. Lines starting with these
        are prefixed with a space to prevent IRC command injection. Default: ."""),
    ),
)
```

**Step 2: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

---

### Task 2: Update sanitization tests for new default

**Files:**
- Modify: `plugins/llm/tests/conftest.py:138`
- Modify: `plugins/llm/tests/test_service.py:341,1342`
- Modify: `plugins/llm/tests/test_stress.py:290`

**Step 1: Update test fixtures**

The test fixtures explicitly set `commandPrefixes` to `[".", "/"]`. Update them to `["."]` to match the new default.

`plugins/llm/tests/conftest.py:138`:
```python
        "commandPrefixes": ["."],
```

`plugins/llm/tests/test_service.py:341`:
```python
            commandPrefixes=["."],
```

`plugins/llm/tests/test_service.py:1342`:
```python
        self.service, self.mock_plugin = make_service(commandPrefixes=["."])
```

`plugins/llm/tests/test_stress.py:290`:
```python
                    "commandPrefixes": ["."],
```

**Step 2: Update the slash sanitization tests**

The existing `test_sanitize_output_slash_prefix` and `test_sanitize_output_multiline_slash` tests expect `/` lines to be sanitized. With the new default, `/` lines should pass through unchanged. Update them:

In `plugins/llm/tests/test_service.py`, find `test_sanitize_output_slash_prefix` (~line 1360):
```python
    def test_sanitize_output_slash_prefix(self) -> None:
        """GIVEN text starting with slash WHEN sanitizing THEN passes through unchanged."""
        text = "/msg someone hello"
        result = self.service.sanitize_output(text)
        assert result == "/msg someone hello"
```

Find `test_sanitize_output_multiline_slash` (~line 1372):
```python
    def test_sanitize_output_multiline_slash(self) -> None:
        """GIVEN multiline text with slash lines WHEN sanitizing THEN passes through unchanged."""
        text = "Line 1\n/quit message\nLine 3"
        result = self.service.sanitize_output(text)
        assert result == "Line 1\n/quit message\nLine 3"
```

Find `test_sanitize_output_mixed_prefixes` (~line 1378):
```python
    def test_sanitize_output_mixed_prefixes(self) -> None:
        """GIVEN multiline text with dots and slashes WHEN sanitizing THEN only dots sanitized."""
        text = ".dot command\n/slash command\nNormal line"
        result = self.service.sanitize_output(text)
        assert result == " .dot command\n/slash command\nNormal line"
```

**Step 3: Run tests**

Run: `make test`
Expected: All PASS

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py \
       plugins/llm/tests/test_service.py plugins/llm/tests/test_stress.py
git commit -m "feat: remove / from default commandPrefixes

Slash-prefixed lines in PRIVMSG are literal text with no protocol-level
risk. Only . (Limnoria command prefix) needs sanitization. This allows
/me action responses to pass through."
```

---

### Task 3: Add system prompt nudge

**Files:**
- Modify: `plugins/llm/src/llm/service.py:296-336`

**Step 1: Write the failing test**

In `plugins/llm/tests/test_service.py`, add a test in the appropriate class (near other `_build_system_prompt` tests, or create a new test if none exist). Search for existing tests of this method first:

```python
def test_build_system_prompt_includes_action_nudge(self, make_service) -> None:
    """GIVEN a base prompt WHEN building system prompt THEN /me nudge is included."""
    service, _ = make_service()
    result = service._build_system_prompt("Be helpful.")
    assert "/me" in result
```

**Step 2: Run test to verify it fails**

Run: `make test -k test_build_system_prompt_includes_action_nudge`
Expected: FAIL (nudge not yet added)

**Step 3: Add the nudge**

In `plugins/llm/src/llm/service.py`, in `_build_system_prompt`, append the nudge after the language instruction block (before the final `return result`):

```python
        result += (
            "\n\nYou may occasionally respond with /me for actions "
            "when it feels natural (e.g., /me shrugs)."
        )

        return result
```

This goes after the `except` block at line 333 and before `return result` at line 336. The final method should end:

```python
        except (AttributeError, KeyError, RuntimeError):
            pass  # Config not available (e.g., in test environment)

        result += (
            "\n\nYou may occasionally respond with /me for actions "
            "when it feels natural (e.g., /me shrugs)."
        )

        return result
```

**Step 4: Run test to verify it passes**

Run: `make test -k test_build_system_prompt_includes_action_nudge`
Expected: PASS

**Step 5: Run full suite**

Run: `make preflight`
Expected: All PASS

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: add /me action nudge to system prompt

Gives the LLM permission to use /me for emotes without modifying
the operator-configurable system prompt."
```

---

### Task 4: Add action detection in ask command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1452-1468`

**Step 1: Write the failing tests**

Add these tests to `plugins/llm/tests/test_commands.py` inside `class TestAskCommand`:

```python
    def test_ask_sends_action_for_me_response(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN sends IRC action."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me shrugs",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mock_action = mocker.patch("llm.plugin.ircmsgs.action")

        plugin.ask(mock_irc, mock_msg, ["how", "are", "you?"])

        mock_irc.reply.assert_not_called()
        mock_action.assert_called_once_with("#test", "shrugs")
        mock_irc.queueMsg.assert_called_once_with(mock_action.return_value)

    def test_ask_normal_response_uses_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds normally WHEN ask called THEN uses irc.reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="The capital is Paris.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["what", "is", "the", "capital?"])

        mock_irc.reply.assert_called_once_with("The capital is Paris.", prefixNick=False)

    def test_ask_action_stores_context_with_star_prefix(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN context stores * BotNick text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me thinks about it",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["hmm"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "* testbot thinks about it"

    def test_ask_action_with_grounding_icon(self, plugin_env, mocker: MockerFixture):
        """GIVEN /me response with grounding WHEN ask called THEN action includes globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me looks it up",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["search", "for", "it"])

        mock_action.assert_called_once_with("#test", "\U0001f310 looks it up")

    def test_ask_bare_me_not_treated_as_action(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with '/me' with no trailing text WHEN ask called THEN uses reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["test"])

        mock_irc.reply.assert_called_once_with("/me", prefixNick=False)
```

**Step 2: Run tests to verify they fail**

Run: `make test -k "test_ask_sends_action or test_ask_normal_response_uses_reply or test_ask_action_stores_context or test_ask_action_with_grounding or test_ask_bare_me"`
Expected: FAIL

**Step 3: Implement action detection**

In `plugins/llm/src/llm/plugin.py`, replace the response handling block in the `ask` method (lines 1452-1464):

```python
                # Format response with grounding icon if search was used
                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                display_response = (
                    f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                )

                # Reply first, then store context (so user gets response even if context fails)
                self.log.info("replying to %s/%s", channel, nick)
                irc.reply(display_response, prefixNick=False)
```

Replace with:

```python
                # Format response with grounding icon if search was used
                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                # Detect /me action prefix
                is_action = response.startswith("/me ") and len(response) > 4
                if is_action:
                    action_text = response[4:]
                    if result.grounding_used:
                        action_text = f"{GROUNDING_ICON} {action_text}"
                    self.log.info("sending action to %s/%s", channel, nick)
                    target = msg.args[0]
                    irc.queueMsg(ircmsgs.action(target, action_text))
                    # Store context as "* BotNick action_text" so follow-ups
                    # understand the bot emoted rather than said something
                    response = f"* {irc.nick} {action_text}"
                else:
                    display_response = (
                        f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                    )
                    self.log.info("replying to %s/%s", channel, nick)
                    irc.reply(display_response, prefixNick=False)
```

**Step 4: Run tests to verify they pass**

Run: `make test -k "test_ask_sends_action or test_ask_normal_response_uses_reply or test_ask_action_stores_context or test_ask_action_with_grounding or test_ask_bare_me"`
Expected: All PASS

**Step 5: Run full suite**

Run: `make preflight`
Expected: All PASS

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat: detect /me in ask responses and send as IRC action

When the LLM starts its response with /me, the bot sends a CTCP
ACTION instead of a regular PRIVMSG. Context stores the action as
'* BotNick text' so follow-up questions understand the emote."
```

---

### Task 5: Final verification

**Step 1: Run full preflight**

Run: `make preflight`
Expected: All checks pass (format, lint, typecheck, tests with 80%+ coverage)

**Step 2: Verify no regressions**

Spot-check that existing ask tests still pass:
Run: `make test -k TestAskCommand -v`
Expected: All existing + new tests pass
