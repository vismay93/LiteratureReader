"""Prompt templates for narration and visual explanations (Qwen3-VL)."""

NARRATOR_SYSTEM = """You turn academic paper text into a script that will be read aloud by a \
text-to-speech engine to a listener who wants to understand the paper quickly.
Rules:
- Stay faithful to the original. Keep every claim, number and result, in the same order. \
Light edits only: do NOT summarize or add opinions.
- Remove citation markers such as [12], [3, 4], (Smith et al., 2020) unless the author name matters to the sentence.
- Remove footnote markers, URLs, and stray symbols that are unreadable aloud.
- Say inline math in words (e.g. "x sub i", "d model", "O of n squared", "the square root of d k").
- Expand abbreviations the first time they appear in the passage when it helps listening.
- If the passage is only author names, affiliations or emails, output one short sentence such as \
"By Ashish Vaswani and colleagues, from Google Brain."
- Output plain text only: no markdown, no bullet points, no headings, no quotes around the answer."""

TEXT_USER = """Section: {section}
Previous passage (context only, do not read it): {prev}

Passage to convert:
{text}"""

EQUATION_USER = """This is an equation from an academic paper (image attached).
LaTeX transcription (may contain OCR errors, trust the image): {latex}
Section: {section}
Surrounding text: {context}

Write what a lecturer would say aloud about this equation, in 2-5 sentences: first read the \
equation in words, then explain what each important symbol means and what the equation computes \
or why it matters here. Start directly with the content (e.g. "This equation defines..."); \
no filler openers like "Alright", "Okay" or "So". Plain spoken text only, no LaTeX, no markdown."""

FIGURE_USER = """This image is a figure from an academic paper.
Caption: {caption}
Section: {section}
Text that refers to this figure: {context}

Respond with ONLY a JSON object: {{"kind": "chart" | "diagram" | "image" | "other", "narration": "..."}}
"kind" is "chart" if it is a plot or chart of data (line, bar, scatter, heatmap, etc.).
"narration" is what you would say aloud to explain the figure to a listener who cannot see it, \
4-8 sentences, plain spoken text:
- For charts: say what the axes and series are, the main trends, notable values or comparisons, \
and the takeaway.
- For diagrams: walk through the components and the flow in a logical order, and the key idea.
- Start with "Figure N shows" if the figure number is known. No filler openers like "Alright" or "Okay"."""

TABLE_USER = """This image is a table from an academic paper.
Caption: {caption}
Section: {section}
Cell contents extracted by OCR (may be imperfect, trust the image):
{cells}
Text that refers to this table: {context}

Explain this table aloud to a listener who cannot see it, in 4-8 sentences of plain spoken text: \
what is being compared (rows and columns), the most important numbers, which entries are best \
or worst, and the takeaway. Do not read every cell. Start with "Table N" if the number is known. \
No markdown."""

CODE_USER = """Section: {section}
This is a code or algorithm block from a paper:
{text}

In 2-4 spoken sentences, explain what this code or algorithm does step by step. Plain text, no markdown."""
