@ECHO OFF
SETLOCAL

IF NOT "%~1"=="" (
    SET ONNX_FILE=%~1
) ELSE (
    SET ONNX_FILE=training_output/latest/optimized.onnx
)
SET UNITY_ASSETS_FOLDER=Assets/Resources/Denoiser

ECHO Publishing %ONNX_FILE% for Unity...

PUSHD %~dp0

python training_script/publish_onnx.py ^
    "%ONNX_FILE%" ^
    --weights-out "%UNITY_ASSETS_FOLDER%/denoiser_weights.bytes" ^
    --json-out "%UNITY_ASSETS_FOLDER%/denoiser_model.json" ^
    --unconsolidated

IF NOT ERRORLEVEL 0 (
    ECHO Failed to publish ONNX model.
    POPD
    EXIT /B %ERRORLEVEL%
)

ECHO Done.
POPD