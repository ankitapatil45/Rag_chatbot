from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
import numpy as np

from semantic.utils.embedding_engine import get_engine

router = APIRouter()


class Question(BaseModel):
    id  : str
    text: str


class SearchRequest(BaseModel):
    questions: List[Question]


class Group(BaseModel):
    group_id : int
    questions: List[Question]


class SearchResponse(BaseModel):
    groups  : List[Group]
    no_match: List[Question]


@router.post("/similar", response_model=SearchResponse)
def search_similar(
    payload  : SearchRequest,
    threshold: float = Query(0.80, ge=0.0, le=1.0, description="Similarity cutoff. Default 0.80"),
):
    engine    = get_engine()
    questions = payload.questions
    n         = len(questions)

    # encode all questions
    vectors = [engine.encode(q.text) for q in questions]

    # union-find to group similar questions
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # compare every pair
    for i in range(n):
        for j in range(i + 1, n):
            score = float(np.dot(vectors[i], vectors[j]))
            if score >= threshold:
                union(i, j)

    # group by root
    from collections import defaultdict
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    groups   = []
    no_match = []
    group_id = 1

    for indices in clusters.values():
        qs = [Question(id=questions[i].id, text=questions[i].text) for i in indices]
        if len(qs) == 1:
            no_match.append(qs[0])
        else:
            groups.append(Group(group_id=group_id, questions=qs))
            group_id += 1

    return SearchResponse(groups=groups, no_match=no_match)