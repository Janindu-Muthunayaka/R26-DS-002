"""
LAYER 4A — title extraction.  OWNER: other team member.

STUB — returns the article unchanged so the pipeline runs end to end.
Replace the body of extract() only. Do not change the signature.
"""
from core.schemas import Article


def extract(img, article: Article) -> Article:
    # TODO (other member): MAT + Tesseract over article.regions
    #                      where label == 'title'
    return article
