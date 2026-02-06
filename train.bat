SET CHECKPOINT_TESTS="checkpoint_tests/exr_easy/*"

python train.py ^
    --input-a-location "training_data/2026-02-04-23-50-02/input_0_*.exr" ^
    --input-b-location "training_data/2026-02-04-23-50-02/input_1_*.exr" ^
    --input-albedo-location "training_data/2026-02-04-23-50-02/albedo_*.exr" ^
    --input-transmissibility-location "training_data/2026-02-04-23-50-02/transmissibility_*.exr" ^
    --reference-location "training_data/2026-02-04-23-50-02/output_*.exr" ^
    --model-path "training_output/model.pth" ^
    --checkpoint-folder "training_output/checkpoints" ^
    --checkpoint-tests %CHECKPOINT_TESTS% ^
    --onnx-export "training_output/model.onnx"
  '  --log-space