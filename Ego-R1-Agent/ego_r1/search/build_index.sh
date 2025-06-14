
corpus_file=example/corpus.jsonl
save_dir=data
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python search_r1/search/index_builder.py \
=======
# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
>>>>>>> 573ed7e86a4143bb13d69752c3a0745cd184bc54
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
