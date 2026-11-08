"""Rule-based narration of plain text (the default; no LLM tokens)."""
import asyncio

import config
from pipeline import narrate
from pipeline.narrate import clean_text, text_source, uses_llm


def test_numeric_citations():
    assert clean_text("Transformers [12] beat RNNs [3, 4] and CNNs [1–5; 9].") == \
        "Transformers beat RNNs and CNNs."


def test_author_year_citations():
    assert clean_text("Attention works well (Vaswani et al., 2017).") == "Attention works well."
    assert clean_text("As shown before (Smith and Jones, 2019; Lee 2021a), it holds.") == \
        "As shown before, it holds."
    assert clean_text("Prior work (e.g., Devlin et al. 2019) used masking.") == "Prior work used masking."
    assert clean_text("Vaswani et al. (2017) introduced it.") == "Vaswani et al. introduced it."


def test_keeps_non_citation_parentheses():
    for s in ["The loss (Eq. 3) is convex.", "Accuracy rose (from 80% to 92%) in 2020.",
              "We use BERT (Bidirectional Encoder Representations from Transformers).",
              "Results on ImageNet (2012 version) improve."]:
        assert clean_text(s) == s


def test_urls_dois_hyphenation_spacing():
    assert clean_text("Code at https://github.com/x/y . See doi:10.1/abc for data.") == \
        "Code at. See for data."
    assert clean_text("a repre- sentation of first- and second-order terms") == \
        "a representation of first- and second-order terms"


def test_text_source_and_llm_use(monkeypatch):
    para = {"type": "text", "text": "A long enough paragraph about attention."}
    authors = {"type": "text", "authors": True, "text": "Ashish Vaswani, Google Brain, a@b.c"}
    fig = {"type": "figure", "caption": "Figure 1: x", "text": ""}
    monkeypatch.setattr(config, "REWRITE_TEXT", False)
    assert text_source(para) == "rules" and not uses_llm(para)
    assert text_source(authors) == "llm" and uses_llm(authors)
    assert uses_llm(fig) and not uses_llm({"type": "heading", "text": "1 Intro"})
    monkeypatch.setattr(config, "REWRITE_TEXT", True)
    assert text_source(para) == "llm" and uses_llm(para)


def test_rules_mode_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr(config, "REWRITE_TEXT", False)

    async def boom(*a, **k):
        raise AssertionError("LLM called for a plain paragraph")
    monkeypatch.setattr(narrate, "chat", boom)
    blocks = [{"type": "text", "text": "Self-attention [7] relates positions (Cheng et al., 2016)."}]
    res = asyncio.run(narrate.narrate_block(None, blocks, 0, None))
    assert res == {"narration": "Self-attention relates positions.", "src": "rules"}
