' Finalize a saved checkpoint model and treat it as the final model.
SET OUTPUT_FOLDER=training_output/latest
SET UNITY_ONNX_FOLDER=Assets/onnx
    
PUSHD %~dp0
python training_script/train_litbox_denoiser.py ^
    --finalize-checkpoint %1 ^
    --output-folder "%OUTPUT_FOLDER%"

python -m onnxsim "%OUTPUT_FOLDER%\final.onnx" "%OUTPUT_FOLDER%\optimized.onnx"
copy "%OUTPUT_FOLDER%\optimized.onnx" "%UNITY_ONNX_FOLDER%\optimized.onnx"
POPD