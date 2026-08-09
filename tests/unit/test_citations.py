"""Tests for citation parsing.

The parser's contract is defined by what the model actually emits, so these
fixtures are shaped like real answers: bracketed markers at sentence ends,
bulleted lists, multi-block citations, and the three failure modes that must
stay distinguishable — supported-but-wrong is verification's problem, but
broken references and uncited claims are visible from the text alone.

No LLM calls anywhere: parsing is pure string work, which is exactly why it
runs on every query while verification does not.
"""

from src.generation.citations import parse_citations


class TestMarkers:
    def test_reads_a_trailing_citation(self):
        parsed = parse_citations("Total net sales were $391.0 billion in fiscal 2024 [2].", 5)

        assert len(parsed.claims) == 1
        assert parsed.claims[0].citations == (2,)
        assert parsed.claims[0].out_of_range == ()

    def test_strips_markers_from_the_claim_text(self):
        """The judge must see the sentence, not the bookkeeping."""
        parsed = parse_citations("Revenue grew 2% [1][3].", 5)

        assert parsed.claims[0].text == "Revenue grew 2%."
        assert parsed.claims[0].raw_text == "Revenue grew 2% [1][3]."

    def test_adjacent_and_comma_separated_forms_agree(self):
        adjacent = parse_citations("Revenue grew two percent [1][3].", 5)
        comma = parse_citations("Revenue grew two percent [1, 3].", 5)

        assert adjacent.claims[0].citations == (1, 3)
        assert comma.claims[0].citations == (1, 3)

    def test_tolerates_the_document_label(self):
        """Models echo the context-block label; answers generated against the
        older '[Document 1]' format still have to parse."""
        parsed = parse_citations("Apple reported record revenue [Document 4].", 5)

        assert parsed.claims[0].citations == (4,)

    def test_repeated_marker_in_one_sentence_counts_once(self):
        parsed = parse_citations("Revenue [2] grew, and margin [2] held.", 5)

        assert parsed.claims[0].citations == (2,)

    def test_mid_sentence_marker_leaves_clean_text(self):
        parsed = parse_citations("Apple [1] reported growth in services.", 5)

        assert parsed.claims[0].text == "Apple reported growth in services."


class TestOutOfRange:
    def test_citation_above_the_block_count_is_out_of_range(self):
        parsed = parse_citations("Net income rose sharply [7].", 5)

        claim = parsed.claims[0]
        assert claim.citations == ()
        assert claim.out_of_range == (7,)
        assert parsed.out_of_range_count == 1

    def test_zero_is_out_of_range_because_blocks_are_one_based(self):
        parsed = parse_citations("Net income rose sharply [0].", 5)

        assert parsed.claims[0].out_of_range == (0,)

    def test_a_broken_reference_is_not_an_uncited_claim(self):
        """The two are different defects — one invented a source, the other
        named none — and lumping them together hides which happened."""
        parsed = parse_citations("Net income rose sharply [7].", 5)

        assert parsed.uncited_claims == ()
        assert parsed.out_of_range_count == 1

    def test_mixed_valid_and_invalid_in_one_claim(self):
        parsed = parse_citations("Revenue rose year over year [2][9].", 5)

        assert parsed.claims[0].citations == (2,)
        assert parsed.claims[0].out_of_range == (9,)


class TestCoverage:
    def test_uncited_claim_lowers_coverage_not_accuracy(self):
        answer = "Revenue was $391.0 billion [1]. Margins also improved notably."
        parsed = parse_citations(answer, 5)

        assert len(parsed.claims) == 2
        assert len(parsed.cited_claims) == 1
        assert len(parsed.uncited_claims) == 1
        assert parsed.coverage == 0.5

    def test_fully_cited_answer_scores_one(self):
        answer = "Revenue was $391.0 billion [1]. Margins improved to 46.2% [2]."

        assert parse_citations(answer, 5).coverage == 1.0

    def test_an_answer_with_no_claims_scores_zero(self):
        """Not 1.0 by vacuous truth — there is nothing attributed here."""
        assert parse_citations("", 5).coverage == 0.0

    def test_citation_pairs_expand_multi_block_claims(self):
        parsed = parse_citations("Revenue rose year over year [1][2].", 5)

        assert parsed.citation_pairs == ((parsed.claims[0], 1), (parsed.claims[0], 2))
        assert parsed.total_citations == 2


class TestSegmentation:
    def test_decimal_points_do_not_split_a_claim(self):
        """Filing prose is full of $1.5 billion and Item 7A. — the shared
        sentence splitter has to survive both."""
        answer = "Revenue was $391.0 billion in fiscal 2024 [1]."
        parsed = parse_citations(answer, 5)

        assert len(parsed.claims) == 1

    def test_bullets_become_claims_without_their_markers(self):
        answer = "Key risks:\n- Supply chain concentration [1]\n- Currency exposure [2]"
        parsed = parse_citations(answer, 5)

        assert [c.text for c in parsed.claims] == [
            "Supply chain concentration",
            "Currency exposure",
        ]

    def test_numbered_list_markers_are_stripped(self):
        answer = "1. Supply chain concentration is material [1]\n2. Currency exposure is hedged [2]"
        parsed = parse_citations(answer, 5)

        assert len(parsed.claims) == 2
        assert parsed.claims[0].citations == (1,)

    def test_an_uncited_heading_is_not_a_claim(self):
        """Otherwise every sectioned answer is marked down for its headings."""
        answer = "Key risks:\n- Supply chain concentration [1]"
        parsed = parse_citations(answer, 5)

        assert len(parsed.claims) == 1
        assert parsed.coverage == 1.0

    def test_a_cited_lead_in_is_a_claim(self):
        answer = "Apple discloses the following risk factors: [1]"
        parsed = parse_citations(answer, 5)

        assert len(parsed.claims) == 1
        assert parsed.claims[0].citations == (1,)

    def test_fragments_that_assert_nothing_are_dropped(self):
        parsed = parse_citations("Yes.\n---\n42\nRevenue rose year over year [1].", 5)

        assert [c.text for c in parsed.claims] == ["Revenue rose year over year."]
