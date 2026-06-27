from llm.verse.reactions import classify_emoji


def test_classify_thumbs_up():
    assert classify_emoji("\U0001f44d") == "approve"


def test_classify_thumbs_down():
    assert classify_emoji("\U0001f44e") == "disapprove"


def test_classify_skin_tone_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d\U0001f3fd") == "approve"  # 👍🏽


def test_classify_variation_selector_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d️") == "approve"  # 👍️


def test_classify_other_emoji_is_other():
    assert classify_emoji("❤️") == "other"  # ❤️


def test_classify_empty_is_other():
    assert classify_emoji("") == "other"
