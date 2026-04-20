  python test.py \
    --config config/sr10_40_vsm_plus.yaml \
    --checkpoint checkpoints/satlib_sr10_40_vsm_plus_finetune/checkpoint_best.pt \
    --test_dir data/satlib_eval_uf75_325/cnf \
    --label_csv data/satlib_eval_uf75_325/labels.csv \
    --batch_size 8
