using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

#region Model Definition

[Serializable]
public class DenoiserModelDefinition
{
    public string[] graph_inputs;
    public string[] graph_outputs;
    public DenoiserOperation[] operations;
}

[Serializable]
public class DenoiserOperation
{
    public string type;
    public string name;
    public string[] inputs;
    public string[] outputs;
    public DenoiserOpParams @params;
}

[Serializable]
public class DenoiserOpParams
{
    public DenoiserWeightInfo weights;
    public DenoiserWeightInfo bias;
    public bool relu;
    public bool pixelShuffle2x2;
    public string pad;
    public string add;
    public string concat;
}

[Serializable]
public class DenoiserWeightInfo
{
    public long offset;
    public int[] shape;
}

#endregion

[RequireComponent(typeof(Simulation))]
public class Denoiser : LitboxComponent
{
    private Simulation simulation;
    private ComputeShader denoiserShader; 

    [Header("Model Files")]
    [SerializeField] private TextAsset modelJson;
    [SerializeField] private TextAsset modelWeights;

    public RenderTexture DenoisedOutput { get; private set; }
    private RenderTexture DenoisedOutputArray { get; set; }

    private DenoiserModelDefinition _model;
    private ComputeBuffer _weightsBuffer;
    private Dictionary<string, RenderTexture> _tensors = new Dictionary<string, RenderTexture>();
    private Dictionary<string, (int c, int h, int w)> _tensorShapes = new Dictionary<string, (int c, int h, int w)>();

    private ComputeShader _inputCompilerShader;

    private int _inputCompilerKernel;
    private int _residualConvKernel;
    private int _maxPoolKernel;

    void OnEnable()
    {
        if (simulation == null) simulation = GetComponent<Simulation>();

        _inputCompilerShader = (ComputeShader)Resources.Load("CompileInputTensor");
        denoiserShader = (ComputeShader)Resources.Load("DenoiserOps");
        if (denoiserShader == null || _inputCompilerShader == null || modelJson == null || modelWeights == null)
        {
            Debug.LogError("Denoiser is missing one or more required assets.");
            enabled = false;
            return;
        }

        LoadModelAndWeights();
        ConsolidateGraph();
        AnalyzeGraph();
        FindKernels();

        simulation.OnStep += OnSimulationStep;
    }

    protected override void OnDisable()
    {
        if (simulation != null)
        {
            simulation.OnStep -= OnSimulationStep;
        }

        _tensors.Clear();
        _tensorShapes.Clear();
    }

    private void LoadModelAndWeights()
    {
        // Parse the model architecture
        _model = JsonUtility.FromJson<DenoiserModelDefinition>(modelJson.text);

        // Load weights into a compute buffer
        // Note: Weights are loaded as raw bytes. The ComputeShader will interpret them as floats.
        _weightsBuffer = new ComputeBuffer(modelWeights.bytes.Length / 4, sizeof(float), ComputeBufferType.Structured);
        _weightsBuffer.SetData(modelWeights.bytes);
        DisposeOnDisable(_weightsBuffer);
    }

    /// <summary>
    /// Consolidates a raw graph of simple operations (BasicConv, Relu, Add, etc.)
    /// into more complex, fused operations like ResidualConv. This mirrors the logic
    /// that was previously in the Python export script.
    /// </summary>
    private void ConsolidateGraph()
    {
        var originalOps = _model.operations;
        var consolidatedOps = new List<DenoiserOperation>();
        var consumedOpNames = new HashSet<string>();

        // Build lookup tables for graph traversal
        var outputToOp = new Dictionary<string, DenoiserOperation>();
        var tensorConsumers = new Dictionary<string, List<DenoiserOperation>>();

        foreach (var op in originalOps)
        {
            foreach (var output in op.outputs)
            {
                outputToOp[output] = op;
            }
            foreach (var input in op.inputs)
            {
                if (!tensorConsumers.ContainsKey(input))
                {
                    tensorConsumers[input] = new List<DenoiserOperation>();
                }
                tensorConsumers[input].Add(op);
            }
        }

        foreach (var op in originalOps)
        {
            if (consumedOpNames.Contains(op.name))
            {
                continue;
            }

            if (op.type == "BasicConv")
            {
                // This is the core of a potential ResidualConv
                var residualConv = new DenoiserOperation
                {
                    type = "ResidualConv",
                    name = op.name,
                    inputs = new List<string>(op.inputs).ToArray(),
                    outputs = new List<string>(op.outputs).ToArray(),
                    @params = new DenoiserOpParams
                    {
                        weights = op.@params.weights,
                        bias = op.@params.bias
                    }
                };
                consumedOpNames.Add(op.name);

                // --- Trace Backwards ---
                string currentInputTensor = op.inputs[0];

                // Check for ClampPad immediately before the Conv
                if (outputToOp.TryGetValue(currentInputTensor, out var prevOpPad) &&
                    prevOpPad.type == "ClampPad" && !consumedOpNames.Contains(prevOpPad.name) &&
                    tensorConsumers.TryGetValue(currentInputTensor, out var padConsumers) && padConsumers.Count == 1)
                {
                    residualConv.@params.pad = "clamp";
                    consumedOpNames.Add(prevOpPad.name);
                    currentInputTensor = prevOpPad.inputs[0];
                }

                // Check for Concat before the (optional) Pad
                if (outputToOp.TryGetValue(currentInputTensor, out var prevOpConcat) &&
                    prevOpConcat.type == "ConcatChannels" && !consumedOpNames.Contains(prevOpConcat.name))
                {
                    residualConv.@params.concat = prevOpConcat.inputs[1];
                    currentInputTensor = prevOpConcat.inputs[0];

                    if (tensorConsumers.TryGetValue(prevOpConcat.outputs[0], out var concatConsumers) && concatConsumers.Count == 1)
                    {
                        consumedOpNames.Add(prevOpConcat.name);
                    }
                }
                residualConv.inputs[0] = currentInputTensor;

                // --- Trace Forwards ---
                string currentOutputTensor = op.outputs[0];
                bool is1x1Conv = op.@params.weights.shape[2] == 1 && op.@params.weights.shape[3] == 1;

                while (tensorConsumers.TryGetValue(currentOutputTensor, out var consumers) && consumers.Count == 1)
                {
                    var nextOp = consumers[0];
                    if (consumedOpNames.Contains(nextOp.name)) break;

                    bool fusedSomething = false;
                    if (!is1x1Conv && nextOp.type == "Add")
                    {
                        var addInputs = new List<string>(nextOp.inputs);
                        addInputs.Remove(currentOutputTensor);
                        residualConv.@params.add = addInputs[0];
                        fusedSomething = true;
                    }
                    else if (nextOp.type == "Relu") { residualConv.@params.relu = true; fusedSomething = true; }
                    else if (nextOp.type == "BasicPixelShuffle") { residualConv.@params.pixelShuffle2x2 = true; fusedSomething = true; }

                    if (fusedSomething)
                    {
                        consumedOpNames.Add(nextOp.name);
                        currentOutputTensor = nextOp.outputs[0];
                    }
                    else { break; }
                }
                residualConv.outputs[0] = currentOutputTensor;
                consolidatedOps.Add(residualConv);
            }
            else if (!consumedOpNames.Contains(op.name))
            {
                var fusableTypes = new HashSet<string> { "Relu", "Add", "ConcatChannels", "ClampPad", "BasicPixelShuffle" };
                if (!fusableTypes.Contains(op.type)) { consolidatedOps.Add(op); }
            }
        }
        _model.operations = consolidatedOps.ToArray();
    }

    private void FindKernels()
    {
        _inputCompilerKernel = _inputCompilerShader.FindKernel("CSMain");
        
        // These kernels need to be implemented in your denoiserShader
        _residualConvKernel = denoiserShader.FindKernel("ResidualConv");
        _maxPoolKernel = denoiserShader.FindKernel("BasicMaxPool");
    }

    /// <summary>
    /// Traverses the model graph to determine the shape of each intermediate tensor.
    /// This is necessary for allocating RenderTextures of the correct size and channel count.
    /// </summary>
    private void AnalyzeGraph()
    {
        _tensorShapes.Clear();
        
        // The input tensor shape is known: 8 channels at simulation resolution.
        var inputShape = (c: 8, h: simulation.height, w: simulation.width);
        _tensorShapes[_model.graph_inputs[0]] = inputShape;

        foreach (var op in _model.operations)
        {
            // This will throw if the graph is disconnected, which is the desired behavior for a malformed model.
            var mainInputShape = _tensorShapes[op.inputs[0]];
            (int c, int h, int w) outputShape = mainInputShape;

            switch (op.type)
            {
                case "ResidualConv":
                    var p = op.@params;

                    // The actual number of input channels for the convolution includes the concatenated tensor.
                    int inputChannels = mainInputShape.c;
                    if (p.concat != null)
                    {
                        // The concat tensor must have been produced by a previous op.
                        if (!_tensorShapes.ContainsKey(p.concat))
                        {
                            throw new KeyNotFoundException($"Shape for concat tensor '{p.concat}' not found. Graph analysis failed.");
                        }
                        inputChannels += _tensorShapes[p.concat].c;
                    }

                    // Sanity check: does the weight tensor match our calculated input channels?
                    if (p.weights.shape[1] != inputChannels)
                    {
                        Debug.LogWarning($"Weight mismatch for op '{op.name}'. Expected {inputChannels} input channels, but weights have {p.weights.shape[1]}.");
                    }

                    // The number of output channels is defined by the weights shape.
                    outputShape.c = p.weights.shape[0];

                    if (p.pixelShuffle2x2)
                    {
                        outputShape.c /= 4; // Depth-to-space with 2x2 blocks reduces channels by 4
                        outputShape.h *= 2;
                        outputShape.w *= 2;
                    }
                    break;

                case "BasicMaxPool":
                    outputShape.h /= 2;
                    outputShape.w /= 2;
                    break;
            }

            foreach (var outputName in op.outputs)
            {
                _tensorShapes[outputName] = outputShape;
            }
        }

        // The internal output from the compute shader is a Tex2DArray
        DenoisedOutputArray = GetOrCreateTensor(_model.graph_outputs[0]);

        // The public-facing output is a regular 2D RenderTexture for material binding
        var shape = _tensorShapes[_model.graph_outputs[0]];
        var desc = new RenderTextureDescriptor(shape.w, shape.h, RenderTextureFormat.ARGBFloat, 0)
        {
            enableRandomWrite = true,
            autoGenerateMips = false
        };
        DenoisedOutput = new RenderTexture(desc);
        DenoisedOutput.name = "DenoisedOutput_Final2D";
        DenoisedOutput.Create();

        DisposeOnDisable(() => DestroyImmediate(DenoisedOutput));
    }

    private RenderTexture GetOrCreateTensor(string name)
    {
        if (_tensors.TryGetValue(name, out var texture))
        {
            return texture;
        }

        if (!_tensorShapes.TryGetValue(name, out var shape))
        {
            throw new KeyNotFoundException($"Shape for tensor '{name}' not found. Graph analysis failed.");
        }

        // Tensors with > 4 channels are stored in a Texture2DArray, where each slice is a float4.
        int channels = shape.c;
        if (channels > 4 && channels % 4 != 0) Debug.LogWarning($"Tensor '{name}' has {channels} channels, which is not a multiple of 4. Packing might be inefficient.");

        int numSlices = (channels + 3) / 4;
        var desc = new RenderTextureDescriptor(shape.w, shape.h, RenderTextureFormat.ARGBFloat, 0)
        {
            enableRandomWrite = true,
            autoGenerateMips = false,
            dimension = UnityEngine.Rendering.TextureDimension.Tex2DArray,
            volumeDepth = numSlices
        };
        var rt = new RenderTexture(desc);
        rt.name = name;
        rt.Create();

        _tensors[name] = rt;
        DisposeOnDisable(() => DestroyImmediate(rt));
        return rt;
    }

    private void OnSimulationStep(int frameCount)
    {
        if (_model == null) return;

        // 1. Compile inputs into the first 8-channel tensor
        var inputTensor = GetOrCreateTensor(_model.graph_inputs[0]);

        // NOTE: This uses the "CompileInputTensor" shader from the older AIAccelerator.
        // That shader was designed to write to a ComputeBuffer (Tensor<float>).
        // This pipeline uses RenderTextures. This call assumes the shader has been adapted
        // to write to a RWTexture2D<float4> named "output". If not, this will fail at runtime.
        _inputCompilerShader.SetVector("size", new Vector2(simulation.width, simulation.height));
        _inputCompilerShader.SetInt("stride", simulation.width * simulation.height);
        _inputCompilerShader.SetTexture(_inputCompilerKernel, "radiance", simulation.SimulationOutputHDR);
        _inputCompilerShader.SetTexture(_inputCompilerKernel, "variance", simulation.VarianceMap);
        _inputCompilerShader.SetTexture(_inputCompilerKernel, "albedo", simulation.GBuffer.AlbedoAlpha);
        _inputCompilerShader.SetTexture(_inputCompilerKernel, "transmissibility", simulation.GBuffer.Transmissibility);
        _inputCompilerShader.SetTexture(_inputCompilerKernel, "output", inputTensor);
        // The input tensor has 8 channels, so it's a Texture2DArray with 2 slices.
        _inputCompilerShader.Dispatch(_inputCompilerKernel, simulation.width / 8, simulation.height / 8, 2);

        // 2. Execute all operations in the graph
        foreach (var op in _model.operations)
        {
            // With a correct JSON, the first input is always a valid tensor.
            RenderTexture input = _tensors[op.inputs[0]];
            RenderTexture output = GetOrCreateTensor(op.outputs[0]);

            // --- Execute ---
            switch (op.type)
            {
                case "ResidualConv":
                    var p = op.@params;

                    // Setup shader keywords for fused operations
                    denoiserShader.SetShaderFlag("USE_RELU", p.relu);
                    denoiserShader.SetShaderFlag("USE_PIXEL_SHUFFLE", p.pixelShuffle2x2);
                    denoiserShader.SetShaderFlag("USE_CLAMP_PAD", p.pad == "clamp");
                    denoiserShader.SetShaderFlag("USE_ADD", p.add != null);
                    denoiserShader.SetShaderFlag("USE_VIRTUAL_CONCAT", p.concat != null);

                    denoiserShader.SetTexture(_residualConvKernel, "_Input", input);
                    denoiserShader.SetTexture(_residualConvKernel, "_Output", output);
                    denoiserShader.SetBuffer(_residualConvKernel, "_Weights", _weightsBuffer);
                    denoiserShader.SetInts("_WeightsShape", p.weights.shape);
                    denoiserShader.SetInt("_WeightOffset", (int)(p.weights.offset / 4)); // Offset in floats
                    denoiserShader.SetInt("_BiasOffset", (int)(p.bias.offset / 4));

                    // Set the number of slices for the main input, needed for virtual concat logic
                    var inputShape = _tensorShapes[op.inputs[0]];
                    int inSlices = (inputShape.c + 3) / 4;
                    denoiserShader.SetInt("_InputSlices", inSlices);

                    if (p.add != null)
                    {
                        denoiserShader.SetTexture(_residualConvKernel, "_AddInput", _tensors[p.add]);
                    }
                    if (p.concat != null)
                    {
                        denoiserShader.SetTexture(_residualConvKernel, "_ConcatInput", _tensors[p.concat]);
                    }
                    
                    int outSlicesConv = (output.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray) ? output.volumeDepth : 1;
                    denoiserShader.Dispatch(_residualConvKernel, (output.width + 7) / 8, (output.height + 7) / 8, outSlicesConv);
                    break;

                case "BasicMaxPool":
                    denoiserShader.SetTexture(_maxPoolKernel, "_Input", input);
                    denoiserShader.SetTexture(_maxPoolKernel, "_Output", output);

                    int outSlicesPool = (output.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray) ? output.volumeDepth : 1;
                    denoiserShader.Dispatch(_maxPoolKernel, (output.width + 7) / 8, (output.height + 7) / 8, outSlicesPool);
                    break;
            }
        }

        // Copy the first slice of the compute output (which is a Tex2DArray)
        // to the public-facing 2D texture.
        Graphics.CopyTexture(DenoisedOutputArray, 0, 0, DenoisedOutput, 0, 0);
    }
}