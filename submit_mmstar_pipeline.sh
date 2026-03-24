#!/bin/bash
set -euo pipefail

PAIR_FILE="pairs.txt"
TMP_FILE="${PAIR_FILE}.tmp"

mkdir -p logs

> "$TMP_FILE"

while IFS= read -r line || [[ -n "$line" ]]; do
    # keep blank lines
    if [[ -z "$line" ]]; then
        echo "" >> "$TMP_FILE"
        continue
    fi

    # keep comment lines
    if [[ "$line" =~ ^# ]]; then
        echo "$line" >> "$TMP_FILE"
        continue
    fi

    # parse columns
    read -r train_jid model_name eval_jid <<< "$line"

    # default eval_jid
    eval_jid="${eval_jid:--}"

    if [[ "$eval_jid" != "-" ]]; then
        echo "[SKIP] ${model_name} already has eval job ${eval_jid}"
        echo "$train_jid $model_name $eval_jid" >> "$TMP_FILE"
        continue
    fi

    echo "======================================"
    echo "Submitting eval for: ${model_name}"
    echo "Depend on training job: ${train_jid}"

    new_jid=$(sbatch --parsable \
        --dependency=afterok:${train_jid} \
        mmstar_infer.sh "${model_name}")

    echo "Submitted eval job: ${new_jid}"

    echo "$train_jid $model_name $new_jid" >> "$TMP_FILE"

done < "$PAIR_FILE"

mv "$TMP_FILE" "$PAIR_FILE"

echo "✅ Updated pairs.txt"