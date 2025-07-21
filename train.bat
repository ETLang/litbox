SET CHECKPOINT_TESTS_EXR="checkpoint_tests/exr_easy/*"
SET CHECKPOINT_TESTS_PNG="checkpoint_tests/png_easy/*"

python train_photoner_3.py ^
    --input-location "training_data/2025-01-20-23-17-48/input_0_*.exr" ^
    --training-location "training_data/2025-01-20-23-17-48/output_*.exr" ^
    --model-path "training_output/model.pth" ^
    --checkpoint-interval 300 ^
    --checkpoint-folder "training_output/checkpoints" ^
    --checkpoint-tests %CHECKPOINT_TESTS_EXR% ^
    --epochs 100 ^
    --crop-size 32 ^
    --upsample 1 ^
    --onnx-export "training_output/model.onnx" ^
    --log-space ^
    --learn-rate 0.00001