' Resume training from a saved checkpoint file.
' Update the path below to the specific .checkpoint file you want to resume from.

SET CHECKPOINT_FILE=training_output/latest/%1.checkpoint
SET UNITY_ONNX_FOLDER=Assets/onnx
    
PUSHD %~dp0
python training_script/train_litbox_denoiser.py ^
    --resume "%CHECKPOINT_FILE%"
@REM    --debug

FOR %%I IN ("%CHECKPOINT_FILE%") DO SET "OUTPUT_FOLDER=%%~dpI"

python -m onnxsim "%OUTPUT_FOLDER%final.onnx" "%OUTPUT_FOLDER%optimized.onnx"
copy "%OUTPUT_FOLDER%optimized.onnx" "%UNITY_ONNX_FOLDER%\optimized.onnx"
POPD