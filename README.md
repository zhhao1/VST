The implementation based on [fairseq](https://github.com/pytorch/fairseq) codebase.


# Pre-processing
We recommend pre-extraction of sentence-level representation features. Due to the large parameters of speech laser model, online extraction often causes OOM problems.
set path
```bash
fairseq=/path/to/fairseq
export PYTHONPATH=${fairseq}:$PYTHONPATH
MUSTC_ROOT=/path/to/must-c
lang=de
pretrain_path=/path/to/speech_laser
pretrain_name=english.pt
```
pre-extraction of sentence-level representation features
```bash
python examples/vae_st/scripts/prepare_utt_pre.py \
  --data-root ${MUSTC_ROOT} --language ${lang} \
  --pretrain-utt-path $pretrain_path --pretrain-utt-name $pretrain_name \
  --process-number 300000 --multi 1
```
pre-processing of must-c data
```bash
python examples/vae_st/scripts/prep_mustc_raw.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type bpe --vocab-size 8000 --language ${lang} 
```

# Training
set path
```bash
w2v_path=/path/to/wav2vec_small.pt
run_dir=${fairseq}/examples/vae_st/run/en-${lang}/run1
SAVE_DIR=${run_dir}/savedir
tensorboard_dir=${run_dir}/tensorboard
```
begin training
```bash
python train.py ${MUSTC_ROOT}/en-${lang}\
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${SAVE_DIR} --num-workers 4 --fp16 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
  --clip-norm 10.0 --seed 1 --update-freq 2 --distributed-world-size 4 \
  --ddp-backend=legacy_ddp \
  --max-tokens 3500000 --max-source-positions 900000 \
  --w2v-path ${w2v_path} \
  --lr 1e-3 --warmup-updates 16000 \
  --patience 5 \
  --tensorboard-logdir ${tensorboard_dir} \
  --best-checkpoint-metric st_loss \
  --word-dropout --word-droprate 0.3 \
  --add-to-embedding --pretrain-utt \
  --vae --hidden-dim 1024 --z-dim 256 --kl-weight 1.0 --kl-annealing-steps 50000 | tee ${run_dir}/log.txt 
```
# Evaluation
average checkpoint
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
```
evaluate the final model 
```bash
python ./fairseq_cli/generate.py ${MUSTC_ROOT}/en-${lang} \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 3500000 --max-source-positions 900000 --beam 5 --scoring sacrebleu
```

