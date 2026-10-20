from rag_eval.client.text import split_reasoning, strip_metadata_header


def test_strip_metadata_header() -> None:
    chunk = (
        "<document_metadata>\nsourceDocument: a.txt\npublished: 10/20/2026, 6:51:29 PM\n"
        "</document_metadata>\n\nThe actual passage."
    )
    assert strip_metadata_header(chunk) == "The actual passage."


def test_strip_metadata_header_leaves_plain_text_alone() -> None:
    assert strip_metadata_header("No header <document_metadata>x</document_metadata>") == (
        "No header <document_metadata>x</document_metadata>"
    )


def test_split_reasoning() -> None:
    assert split_reasoning("<think>\nLet me check.\n</think>The answer is 42.") == (
        "Let me check.",
        "The answer is 42.",
    )


def test_split_reasoning_without_think_block() -> None:
    assert split_reasoning("  Plain answer. ") == (None, "Plain answer.")


def test_split_reasoning_only_strips_a_leading_block() -> None:
    text = "Answer first <think>not reasoning</think>"
    assert split_reasoning(text) == (None, text)
