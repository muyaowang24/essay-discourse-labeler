from essay_labeler.labels import DEFAULT_LABELS, discourse_to_bio, id_to_label, label_to_id


def test_label_mappings_are_inverse() -> None:
    label_ids = label_to_id(DEFAULT_LABELS)
    restored = id_to_label(DEFAULT_LABELS)
    assert restored[label_ids["B-Lead"]] == "B-Lead"


def test_discourse_to_bio_marks_beginning_and_inside() -> None:
    entities = discourse_to_bio("Claim", [2, 3, 4], total_tokens=6)
    assert entities == ["O", "O", "B-Claim", "I-Claim", "I-Claim", "O"]
