' Generate training data by opening Training.scene in Unity.
' Make sure 'train' is checked on the traininer manager GameObject and just press play.
' A mobile Geforce 3080 can make about 3600 training images a day.
' Make sure you note which folder the generated data gets saved to
' (it's different for every session, unless 'Continue Previous Session' is checked)
' and update the path below.

SET TRAINING_DATA_FOLDER=training_data/2026-02-08-01-17-30
SET OUTPUT_FOLDER=training_output/latest
SET UNITY_ONNX_FOLDER=Assets/onnx
SET SKIP_CACHE_VALIDATION=True


if "%SKIP_CACHE_VALIDATION%"=="True" (
  SET ARG_SKIP_CACHE_VALIDATION=--skip-cache-validation
) else (
  SET ARG_SKIP_CACHE_VALIDATION=
)
    
PUSHD %~dp0
python training_script/train_litbox_denoiser.py ^
    --input-a-location "%TRAINING_DATA_FOLDER%/Input5_Radiance_A_*.exr" ^
    --input-b-location "%TRAINING_DATA_FOLDER%/Input5_Radiance_B_*.exr" ^
    --input-albedo-location "%TRAINING_DATA_FOLDER%/Albedo_*.png" ^
    --input-density-location "%TRAINING_DATA_FOLDER%/Density_*.exr" ^
    --reference-location "%TRAINING_DATA_FOLDER%/Output_Reference_*.exr" ^
    --output-folder "%OUTPUT_FOLDER%" ^
    --test-ratio 0.1 ^
    --epochs 25 ^
    %ARG_SKIP_CACHE_VALIDATION%
@REM    --debug

@REM @IF NOT ERRORLEVEL 0 (
@REM     POPD
@REM     EXIT /B %ERRORLEVEL%
@REM )

python -m onnxsim "%OUTPUT_FOLDER%\final.onnx" "%OUTPUT_FOLDER%\optimized.onnx"
copy "%OUTPUT_FOLDER%\optimized.onnx" "%UNITY_ONNX_FOLDER%\optimized.onnx"
POPD