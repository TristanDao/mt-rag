# semetic_router/route.py

from Products_RAG_main.semetic_router.chitchat_router import ChitchatRouter
from Products_RAG_main.semetic_router.samples import chitchatSample

FACT_KEYWORDS = [
    "giá",
    "bao nhiêu",
    "hôm nay",
    "tăng",
    "giảm",
    "lãi suất",
    "usd",
    "vàng",
    "xăng"
]


def is_fact_query(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FACT_KEYWORDS)



class SemanticRouteDecision:
    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold
        self.router = ChitchatRouter(chitchatSample)

    def decide(self, query: str) -> dict:
    # # 1️⃣ RULE-BASED FACT QUERY (CHẶN SỚM)
    #     if is_fact_query(query):
    #         return {
    #             "is_chitchat": is_chitchat,
    #             "similarity": 1.0,
    #             "use_retrieval": not is_chitchat,
    #             "reason": "fact_query_rule"
    #         }

        # 2️⃣ SEMANTIC SIMILARITY (CHITCHAT)
        score = self.router.similarity(query)
        is_chitchat = score >= self.threshold

        return {
            "is_chitchat": is_chitchat,
            "similarity": score,
            "use_retrieval": not is_chitchat,
            "reason": "semantic_similarity"
        }