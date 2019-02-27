python $HOME/inequity-release/inequity/train.py \
    --augmented_loss_weights [1,1,1,1] \
    OUTPUT_DIR /root/inequity-release/weights/weighted1 \
    MODEL.WEIGHT "catalog://ImageNetPretrained/MSRA/R-50" \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.01 \
    SOLVER.MAX_ITER 3350 \
    SOLVER.STEPS "(2233,2792)" \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.IMS_PER_BATCH 1 &
 