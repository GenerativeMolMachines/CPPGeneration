from Bio import pairwise2


def calculate_alignment_similarity(sequence1, sequence2, alignment_type="global", match_score=2, mismatch_penalty=-1, gap_penalty=-0.5, extension_penalty=-0.1):
    """
    Calculates the percentage similarity from pairwise alignment of two sequences (DNA or protein).

    Args:
        sequence1 (str): The first sequence (DNA or protein).
        sequence2 (str): The second sequence.
        alignment_type (str): Alignment type: "global" (Needleman-Wunsch) or "local" (Smith-Waterman). Default value: "global".
        match_score (int): Score for matching characters. Default value: 2.
        mismatch_penalty (int): Penalty for mismatched characters. Default value: -1.
        gap_penalty (float): Penalty for a gap. Default value: -0.5.
        extension_penalty (float): Penalty for gap extension. Default value: -0.1.

    Returns:
        float: Percentage of matching characters in the aligned pairwise sequences (between 0 and 100). Returns 0 if alignment fails.
    """

    if alignment_type not in ["global", "local"]:
        raise ValueError("alignment_type must be 'global' or 'local'")

    if not sequence1 or not sequence2:
        return 0.0

    try:
        if alignment_type == "global":
            # Global alignment (Needleman-Wunsch algorithm)
            alignments = pairwise2.align.globalms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
        else:
            # Local alignment (Smith-Waterman algorithm)
            alignments = pairwise2.align.localms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
    except Exception as e:
        print(f"Alignment error: {e}")
        return 0.0

    if not alignments:
        return 0.0

    # Take the best alignment (the first one)
    alignment1, alignment2, score, begin, end = alignments[0]

    # Count matching characters
    matches = sum(1 for i in range(len(alignment1)) if alignment1[i] == alignment2[i] and alignment1[i] != '-')

    # Total alignment length (including gaps)
    alignment_length = len(alignment1)

    # Similarity percentage
    similarity_percentage = (float(matches) / alignment_length) * 100 if alignment_length > 0 else 0.0

    return similarity_percentage