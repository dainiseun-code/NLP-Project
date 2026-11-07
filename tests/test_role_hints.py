"""Fictional syntax-only role-hint checks; this is not recipient selection."""

from messaging_nlp.role_hints import role_hints, single_addressed_contact

CONTACTS = [
    {"contact_id": "c1", "name": "Maya Chen", "aliases": ["May", "Maya"]},
    {"contact_id": "c2", "name": "Maya Patel", "aliases": ["MP", "Maya"]},
    {"contact_id": "c3", "name": "Alex Morgan", "aliases": ["AM", "Alex"]},
    {"contact_id": "c4", "name": "Theo Nguyen", "aliases": ["T"]},
]


def test_addressed_name_is_separate_from_subject_name():
    hints = role_hints(
        "Please find out from Maya Patel whether Alex Morgan plans to attend.",
        CONTACTS,
    )
    assert [item["contact_id"] for item in hints["addressed"]] == ["c2"]
    assert [item["contact_id"] for item in hints["subject_only"]] == ["c3"]
    assert hints["unresolved_identity"] is False


def test_shared_alias_is_unresolved_without_a_full_name():
    hints = role_hints("Ask Maya a quick question.", CONTACTS)
    assert {item["contact_id"] for item in hints["addressed"]} == {"c1", "c2"}
    assert hints["unresolved_identity"] is True


def test_full_name_shadows_the_shared_alias():
    hints = role_hints("Ask Maya Patel a quick question.", CONTACTS)
    assert [item["contact_id"] for item in hints["addressed"]] == ["c2"]
    assert hints["unresolved_identity"] is False


def test_short_alias_does_not_match_a_contraction():
    hints = role_hints("Let Maya Patel know I won't be there.", CONTACTS)
    assert [item["contact_id"] for item in hints["addressed"]] == ["c2"]
    assert all(item["contact_id"] != "c4" for item in hints["addressed"])


def test_single_addressed_contact_refuses_ambiguity():
    assert single_addressed_contact("Ask Maya Patel a question.", CONTACTS) == "c2"
    assert single_addressed_contact("Ask Maya a question.", CONTACTS) is None


def test_duplicate_full_names_are_unresolved():
    contacts = [
        {"contact_id": "a", "name": "Sam Lee", "aliases": ["Sam"]},
        {"contact_id": "b", "name": "Sam Lee", "aliases": ["Sam"]},
    ]
    hints = role_hints("Ask Sam Lee a question.", contacts)
    assert hints["unresolved_identity"] is True
    assert single_addressed_contact("Ask Sam Lee a question.", contacts) is None


def test_draft_to_and_asking_are_address_cues():
    first = role_hints("Draft a note to Alex about the visit.", CONTACTS)
    second = role_hints("Draft a note asking Alex about the visit.", CONTACTS)
    assert [item["contact_id"] for item in first["addressed"]] == ["c3"]
    assert [item["contact_id"] for item in second["addressed"]] == ["c3"]
