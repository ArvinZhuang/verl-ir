
dataset=dl20
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-passage \
  --topics ${dataset}-passage \
  --output runs/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
  --bm25 --k1 0.9 --b 0.4


python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.100,1000 ${dataset}-passage \
  runs/run.msmarco-v1-passage.bm25-default.${dataset}.txt

python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-passage \
  --topics /scratch3/zhu042/verl-ir/generate_data/hyde-bm25/${dataset}.expanded.tsv \
  --output runs/run.msmarco-v1-passage.hyde-bm25.${dataset}.txt \
  --bm25 --k1 0.9 --b 0.4

python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.100,1000 ${dataset}-passage \
  runs/run.msmarco-v1-passage.hyde-bm25.${dataset}.txt
