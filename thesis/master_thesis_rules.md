# PRISMA Lab's master thesis writing checklist

*Transcribed from `master_thesis_rules.pdf` (PRISMA Lab, June 11, 2026) — the official guide for writing the thesis in LaTeX. Governs `technical_report.tex`.*

## Thesis structure

No single structure fits everyone; adapt this outline as needed:

1. Abstract
2. Introduction
   a. Motivation
   b. Contributions
3. State of the art
4. Methodology
5. Experiments and results
   a. Setup of the experiments
   b. Results
6. Discussion
7. Conclusions
8. References

## Writing style

- Use consistent tense — present, usually, unless reporting results achieved in earlier papers.
- Parentheses or brackets are always surrounded by a space.
  - ✗ `The experiment(Fig. 7)shows`
  - ✓ `The experiment (Fig. 7) shows`
- Abbreviations and acronyms are always explained before use. Expand all acronyms on first use, e.g., Asynchronous Transfer Mode (ATM). Do not use abbreviations in the title or heads unless unavoidable.
- In formal writing, contractions like "don't", "doesn't", "won't", "it's" are generally avoided.
- "i.e." means "that is"; "e.g." means "for example". Both are always followed by a comma.
- Do not use the abbreviation "w.r.t." for "with respect to".
- "Therefore", "however", "hence", "thus" are usually followed by a comma. ✓ "Therefore, our idea should not be implemented."
- Do not use "essentially" to mean "approximately" or "effectively".
- "On the other hand, " should always be preceded by "On one hand, ".
- Use the serial (Oxford) comma in 3+ word series. ✓ "Red, yellow, and blue wires" — ✗ "Red, yellow and blue wires".
- Page titles: title case (words ≥4 letters capitalized, plus nouns/pronouns/verbs/adjectives; hyphenated words capitalize after the hyphen, e.g., "Pre-University"). Section headers, form fields, and other page content: sentence case — only proper nouns and titles capitalized.
- *Never* write code inside the thesis. Use pseudocode to describe an algorithm if needed.

## Hyphens and dashes

Three distinct dashes — use the correct one:

- **Hyphen (`-`)**: compound words / compound adjectives. ✓ "twenty-six birds", ✓ "high-performance implementation".
- **En dash (`–`)**: ranges of numbers, and to connect words that are separate but function together.
  - ✗ `Lines 1-3 present our results.` / ✗ `Lines 1 − 3 present our results.` → ✓ `Lines 1–3 present our results.`
  - ✗ `Human-robot interaction.` → ✓ `Human–robot interaction.`
- **Em dash (`—`)**: replaces commas, parentheses, colons, or semicolons. ✓ "Alexander the Great—a king among kings—led a nearly unstoppable army."

## Writing units

- Nonbreaking space between measurement and unit (`~` in LaTeX); unit not italicized. ✗ `10m` / ✗ `10m` (italic) → ✓ `10~m`.
- Use a leading zero before decimal points: "0.25", not ".25".

## Referencing

- Make proper use of `\label{}`, `\ref{}`, `\eqref{}`, `\cite{}`, etc. so the reader can navigate.
- Use "(1)", not "Eq. (1)" or "equation (1)", except at the start of a sentence: "Equation (1) is …" — use `\eqref{}` there (it includes the brackets).
- Use "in Figure 1", not "the following figure" (figures may move during typesetting).
- "Section", "Figure", "Table" are capitalized (e.g., "As discussed in Section 3"). "Figure" may abbreviate to "Fig."; the others usually aren't — just be consistent.
- Tie figure number to its reference with a nonbreaking space: `Fig.~\ref{fig:arch}`.
- Do not refer to colors in graphs (monochrome printing; ~1 in 12 men and 1 in 200 women are color blind). Use color *and* shape/line-style together, with a legend; connect data labels to lines rather than relying on a color key (IEEE guidance).

## Equations

- Equations are part of the text — punctuate accordingly. Don't introduce an equation with a colon; follow it with a comma (phrase continues) or full stop (phrase ends).
- Define every symbol before or immediately after the equation (meaning + belonging set), in text or in a table. Don't reintroduce a symbol already defined.
  - ✗ "Let us introduce the example equation: `a + b = γ` (1)"
  - ✓ "Let us introduce the example equation `a + b = γ,` (2) with `a, b, γ ∈ R` example symbols."
- All equations referenced in the text must be numbered.
- Typeset negative numbers in math mode (`−1`), not with a hyphen (`-1`).
- Scalars: lowercase, not bold. Vectors: lowercase, can be bold. Matrices: uppercase, can be bold.
- Bold usage for vectors/matrices must be coherent throughout the whole document — all bold, or none.
- Math accents (`\dot`, `\bar`, `\tilde`, `\hat`, …) must not include subscripts/superscripts under the accent. ✗ `\dot{x_e}` → ✓ `\dot{x}_e`.
- Subscripts and superscripts must not be bold. ✗ `\mathbf{x_e}` (bold subscript) → ✓ `x_{\mathbf{e}}`-style — i.e., keep sub/superscripts non-bold even if the base symbol is bold.

## Figures & tables

- All graphs/schemes/plots must be vector graphics (SVG, EPS, PDF export). Avoid raster formats (PNG, JPEG) unless strictly necessary (e.g., setup photo/screenshot) — and then high-resolution only.
- Positioning: `[t]` for figures (top of page), `[b]` for tables (bottom). Avoid mid-column placement.
- Captions are mandatory and must clearly explain the figure — specify every symbol and every line/series. End captions with a full stop; never a colon.
- Distinguish graph lines with both color and style (solid/dots/dashes/mix); avoid red, green, yellow — prefer blue and orange for contrast/visibility. Always include a legend, positioned so it doesn't overlap plot content.
- Illustration/scheme/plot backgrounds must be transparent or white.
- Size figures and font so no in-document rescaling is needed; any in-figure text must be easily readable at the coherent font size.
- Avoid serif fonts in plots (poor legibility). Prefer LaTeX-matching fonts for labels/tick marks.

## References (bibliography)

- Never use citations as a sentence's subject/word.
  - ✗ "[15] introduce a similar strategy" / ✗ "A similar strategy is introduced in [15]"
  - ✓ "Smith et al. introduce a similar strategy [15]"
- Introduce citations with a nonbreaking space: `text~\cite{Foo:2000:BAR}`.
- Build the bibliography via a thorough literature search (start from advisor's material, then Google Scholar); cluster related papers when citing; close the state-of-the-art section by comparing it against your own contribution.
- Fix BibTeX capitalization by bracing words that must stay capitalized in titles, e.g., `{GPU}`; make sure every field is filled and readable.
- Use consistent title capitalization across all references — all title-case or all sentence-case, not mixed.
- Prefer up-to-date references — check whether a cited draft/technical-report/workshop paper was later superseded by a conference or journal version.
- Author names consistent throughout: either all full names (John Doe) or all abbreviated (J. Doe) — never mixed.
- The citation must make it obvious whether a work is a conference or journal paper.
  - Conference references need: location, month and year, conference name/proceedings, pages.
  - Journal references need: volume, issue number, pages, journal name.

## Other

- Unless explicitly requested, don't share thesis source files — share the PDF. Name it `<NAME>_<SURNAME>_master_thesis.pdf`.
- If source sharing is explicitly requested (e.g., via Overleaf), name the repo/folder `<NAME>_<SURNAME>_master_thesis`.
- Keep the final PDF under 20 MB.
- Only send the thesis once it's in final form; use motivated placeholders for anything genuinely missing (e.g., a pending figure).
