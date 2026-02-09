' Generate training data by opening Training.scene in Unity.
' Make sure 'train' is checked on the traininer manager GameObject and just press play.
' A mobile Geforce 3080 can make about 3600 training images a day.
' Make sure you note which folder the generated data gets saved to
' (it's different for every session, unless 'Continue Previous Session' is checked)
' and update the path below.

SET TRAINING_DATA_FOLDER=training_data/2026-02-08-01-17-30
SET CHECKPOINT_TESTS="checkpoint_tests/exr_easy/*"

python training_script/train_litbox_denoiser.py ^
    --input-a-location "%TRAINING_DATA_FOLDER%/input_0_*.exr" ^
    --input-b-location "%TRAINING_DATA_FOLDER%/input_1_*.exr" ^
    --input-albedo-location "%TRAINING_DATA_FOLDER%/albedo_*.exr" ^
    --input-transmissibility-location "%TRAINING_DATA_FOLDER%/transmissibility_*.exr" ^
    --reference-location "%TRAINING_DATA_FOLDER%/output_*.exr" ^
    --model-path "training_output/model.pth" ^
    --checkpoint-folder "training_output/checkpoints" ^
    --checkpoint-tests %CHECKPOINT_TESTS% ^
    --onnx-export "training_output/model.onnx"
  '  --log-space