from pipeline.narrate import heading_narration, visual_label, find_reference_context
from pipeline.tts import split_sentences


def test_heading():
    assert heading_narration("3.2 Attention") == "Section 3.2. Attention."
    assert heading_narration("Abstract") == "Abstract."


def test_split_sentences_abbrev():
    s = split_sentences("We use e.g. dropout. See Fig. 2 for details. Results improve.")
    assert s == ["We use e.g. dropout.", "See Fig. 2 for details.", "Results improve."]


def test_reference_context():
    blocks = [{"type": "text", "text": "As shown in Fig. 3, it works."},
              {"type": "figure", "caption": "Figure 3: Plot.", "text": ""}]
    assert visual_label(blocks[1]) == ("Figure", "3")
    assert "Fig. 3" in find_reference_context(blocks, visual_label(blocks[1]), 1)
