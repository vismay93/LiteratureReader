from pipeline import doctags

PAGE = (
    "<doctag><page_header><loc_10><loc_5><loc_200><loc_12>arXiv preprint</page_header>"
    "<section_header_level_1><loc_40><loc_60><loc_200><loc_70>1 Introduction</section_header_level_1>"
    "<text><loc_40><loc_75><loc_240><loc_120>Transformers [12] are great.</text>"
    "<picture><loc_260><loc_60><loc_460><loc_200><caption><loc_260><loc_205><loc_460><loc_220>"
    "Figure 1: The model.</caption></picture>"
    "<formula><loc_40><loc_130><loc_240><loc_145>E = mc^2</formula>"
    "<otsl><loc_40><loc_300><loc_460><loc_400><ched>Model<ched>BLEU<nl><fcel>Base<fcel>27.3<nl></otsl>"
    "<caption><loc_40><loc_402><loc_460><loc_412>Table 2: Results.</caption>"
    "<page_footer><loc_240><loc_490><loc_260><loc_495>3</page_footer>"
    "<section_header_level_1><loc_40><loc_420><loc_200><loc_430>References</section_header_level_1>"
    "<text><loc_40><loc_432><loc_460><loc_480>[1] A. Author. A paper.</text></doctag>"
)


def test_parse_types_order_bbox():
    blocks = doctags.parse_page(PAGE, 0)
    types = [b["type"] for b in blocks]
    assert types == ["page_header", "heading", "text", "figure", "equation", "table",
                     "caption", "page_footer", "heading", "text"]
    assert blocks[1]["bbox"] == [0.08, 0.12, 0.4, 0.14]
    assert blocks[3]["caption"] == "Figure 1: The model."
    assert blocks[4]["latex"] == "E = mc^2"
    assert blocks[5]["table"] == [["Model", "BLEU"], ["Base", "27.3"]]


def test_postprocess():
    blocks = doctags.postprocess(doctags.parse_page(PAGE, 0))
    types = [b["type"] for b in blocks]
    assert "page_header" not in types and "page_footer" not in types
    table = next(b for b in blocks if b["type"] == "table")
    assert table["caption"] == "Table 2: Results."
    cap = next(b for b in blocks if b["type"] == "caption")
    assert cap["skip"]
    refs = blocks[-1]
    assert refs["skip"] and refs["skip_reason"] == "references"
    assert blocks[1]["section"] == "1 Introduction"
    assert [b["id"] for b in blocks][:2] == ["p0b0", "p0b1"]


def test_malformed_tolerated():
    blocks = doctags.parse_page("<text><loc_1><loc_2><loc_3><loc_4>unterminated", 0)
    assert blocks[0]["text"] == "unterminated"
    assert doctags.parse_page("<text>no locs</text>", 0) == []


def test_chart_subtype():
    b = doctags.parse_page("<picture><loc_1><loc_2><loc_300><loc_300><line_chart></picture>", 0)
    assert b[0]["type"] == "chart"
