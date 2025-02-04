# https://github.com/aflah02/HubbleSuite/blob/ameya-dev/scripts/tokenize_dclm.sh

set -e
set -x

NEOX_DIR=/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox
DATA_DIR=/NS/llm-artifacts/static00/dclm-baseline-1.0
VOCAB_DIR=/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts

exp_name='Tokenized_DCLM'
json_dataset="$(find ${DATA_DIR}/datasets/global-shard_01_of_10/local-shard_1_of_10/ -type f -print0 | sort -z | tr '\0' ',')"
tokenized_dir="${DATA_DIR}/${exp_name}/tokenized"

mkdir -p $tokenized_dir/hubble-copyright

log_file="/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/Tokenization_Logs/${exp_name}-hubble-v2-1-tokenize_copyright_data.log"
for filepath in "/home/ameya/datasets/hubble-v2-1/"*_dup.jsonl
do
      echo "Processing file: $filepath"
      filename=$(basename "$filepath")
      fileprefix="${filename%.jsonl}"
      python $NEOX_DIR/tools/datasets/preprocess_data.py \
            --input "/home/ameya/datasets/hubble-v2-1/${filename}" \
            --output-prefix "$tokenized_dir"/hubble/${fileprefix} \
            --vocab ${VOCAB_DIR}/olmo_tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 16 2>&1 | tee -a ${log_file}
done