#!/bin/bash
# 消融实验批量运行脚本 (DDP 多卡训练)
# 遍历 7 个配置 × 3 个种子，依次执行 train_ddp + test

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIGS=(
    "config/ablation/A0_full.yaml"
    "config/ablation/A1_no_multichannel.yaml"
    "config/ablation/A2_no_negation.yaml"
    "config/ablation/A3_full_attention.yaml"
    "config/ablation/A4_mean_pool.yaml"
    "config/ablation/A5_no_polarity.yaml"
    "config/ablation/A6_no_core_loss.yaml"
)

NAMES=(
    "A0_full"
    "A1_no_multichannel"
    "A2_no_negation"
    "A3_full_attention"
    "A4_mean_pool"
    "A5_no_polarity"
    "A6_no_core_loss"
)

SEEDS=(42)

TRAIN_DATA="${TRAIN_DATA:-data_test_3_10/train.pt}"
VAL_DATA="${VAL_DATA:-data_test_3_10/val.pt}"
TEST_DATA="${TEST_DATA:-data_test_3_10/test.pt}"
RESULT_DIR="results/ablation"

# DDP 多卡配置: 默认使用所有可见 GPU
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"

mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "GeoSATformer v2 Ablation Study (DDP: ${NGPUS} GPUs)"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "  Total runs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"
echo "=============================================="

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${NAMES[$i]}"

    for seed in "${SEEDS[@]}"; do
        run_name="${name}_seed${seed}"
        log_file="${RESULT_DIR}/${run_name}.log"
        ckpt_dir="checkpoints/ablation/${name}"
        log_dir="logs/ablation/${name}_seed${seed}"

        echo ""
        echo ">>> [$run_name] Training..."
        echo "    Config: $config"
        echo "    Seed: $seed"
        echo "    Log: $log_file"

        # 训练 (DDP 多卡)
        torchrun --nproc_per_node="$NGPUS" --master_port=29500 train_ddp.py \
            --config "$config" \
            --train_path "$TRAIN_DATA" \
            --val_path "$VAL_DATA" \
            --seed "$seed" \
            2>&1 | tee "${RESULT_DIR}/${run_name}_train.log"

        # 测试
        echo ">>> [$run_name] Testing..."
        python test.py \
            --config "$config" \
            --checkpoint "${ckpt_dir}/checkpoint_best.pt" \
            --test_path "$TEST_DATA" \
            2>&1 | tee "${RESULT_DIR}/${run_name}_test.log"

        echo ">>> [$run_name] Done."
    done
done

echo ""
echo "=============================================="
echo "All ablation experiments completed!"
echo "Results saved to: $RESULT_DIR"
echo ""
echo "Run analysis: python scripts/analyze_ablation.py"
echo "=============================================="
