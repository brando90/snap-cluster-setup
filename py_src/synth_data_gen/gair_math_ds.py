"""
{
    "text": ...,
    "SubSet": "CommomCrawl" | "StackExchange" | "Textbooks" | "Wikipedia" | "ProofWiki" | "arXiv"
    "meta": {"language_detection_score": , "idx": , "contain_at_least_two_stop_words": ,
}
"""
"""
# -- Downloading the dataset
mkdir gair
huggingface-cli download --resume-download --repo-type dataset GAIR/MathPile --local-dir /lfs/skampere1/0/brando9/data/gair --local-dir-use-symlinks False

# IDK if needed
# $ cd /lfs/skampere1/0/brando9/data
# $ find . -type f -name "*.gz" -exec gzip -d {} \;
"""
from datasets import load_dataset

dataset = load_dataset()