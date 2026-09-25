from poc.agent.citations import cited_chunk_ids
from poc.agent.nodes.judge import _extract_citations


def test_single_and_grouped_citations():
    text = "Grip is great [a__0__c0]. Runs small [a__1__c0, a__2__c0; a__0__c0]."
    assert cited_chunk_ids(text) == ["a__0__c0", "a__1__c0", "a__2__c0"]


def test_no_citations():
    assert cited_chunk_ids("plain text") == []


def test_judge_keeps_only_retrieved_ids_from_grouped_brackets():
    retrieved = [
        {"chunk_id": "a__0__c0", "text": "grip", "score": 0.9},
        {"chunk_id": "a__1__c0", "text": "size", "score": 0.8},
    ]
    citations = _extract_citations("x [a__0__c0, a__1__c0, zz__9__c0] [see note]", retrieved)
    assert [c.chunk_id for c in citations] == ["a__0__c0", "a__1__c0"]
