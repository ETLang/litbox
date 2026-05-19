using System;
using System.Linq;
using System.Text;
using GLTFast.Schema;
using Unity.VisualScripting;
using UnityEngine;

public class AIAccelerator : MonoBehaviour {
    [SerializeField] private Simulation simulation;
    [SerializeField] private Unity.InferenceEngine.ModelAsset accelerationModel;
    [SerializeField] private bool operateOnToneMapped;

    public RenderTexture HDROutputTexture { get; private set; }
    public RenderTexture ToneMappedOutputTexture { get; private set; }

    Unity.InferenceEngine.Worker aiWorker;
    Unity.InferenceEngine.Tensor<float> sourceTensor;
    ComputeShader _inputCompilerShader;
    int _inputCompilerKernel;

    void Start() {
        if(simulation) {
            simulation.OnStep += Simulation_OnStep;
            HDROutputTexture = new RenderTexture(simulation.width, simulation.height, 0, RenderTextureFormat.ARGBFloat);
            HDROutputTexture.Create();
            ToneMappedOutputTexture = new RenderTexture(simulation.width, simulation.height, 0, RenderTextureFormat.ARGB32);
            ToneMappedOutputTexture.Create();
        }

        var model = Unity.InferenceEngine.ModelLoader.Load(accelerationModel);
        aiWorker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.GPUCompute);
        
        // Analyze layer dimensions
        var layerOutputShapes = new Unity.InferenceEngine.TensorShape[model.layers.Count];
        var currentShape = new Unity.InferenceEngine.TensorShape(1, 8, simulation.height, simulation.width);
        
        var sb = new StringBuilder();

        /*

        Relevant Types:
        Pad
        Conv
        Add
        Relu
        MaxPool
        DepthToSpace
        Concat

        */
        var relevantTypes = model.layers.Select(l => l.GetType()).Distinct();
        sb.AppendLine("Relevant Types:");
        foreach(var t in relevantTypes)
        {
            sb.AppendLine(t.Name);
        }
        sb.AppendLine();


        for (int i = 0; i < model.layers.Count; i++)
        {
            var layer = model.layers[i];
            // We reason that the output shape depends on the layer type (Conv, Pooling, etc.)
            // The Inference Engine's internal logic determines this during execution, 
            // but we track the progression here for analysis.
            var layerType = layer.GetType();

            sb.AppendLine($"{i} - {layerType}");

            //layerOutputShapes[i] = model.GetLayerShape(layer.name);
        }

        Debug.Log(sb);

        _inputCompilerShader = (ComputeShader)Resources.Load("CompileInputTensor");
        _inputCompilerKernel = _inputCompilerShader.FindKernel("CSMain");

        _inputCompilerShader.SetVector("size", new Vector2(simulation.width, simulation.height));
        _inputCompilerShader.SetInt("stride", simulation.width * simulation.height);
    }

    void OnDisable() {
        if(simulation) {
            simulation.OnStep -= Simulation_OnStep;  
        }

        if(aiWorker != null) {
            aiWorker.Dispose();
            aiWorker = null;
        }

        if(sourceTensor != null) {
            sourceTensor.Dispose();
            sourceTensor = null;
        }

        if(HDROutputTexture != null) {
            DestroyImmediate(HDROutputTexture);
            HDROutputTexture = null;
        }

        if(ToneMappedOutputTexture) {
            DestroyImmediate(ToneMappedOutputTexture);
            ToneMappedOutputTexture = null;
        }
    }

    void Simulation_OnStep(int frameCount) {
        // 8 Channels: Radiance (3), Variance (1), Albedo (3), Density (1)
        sourceTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 8, simulation.height, simulation.width));

        Unity.InferenceEngine.Tensor<float> outputTensor = null;

        // Combine Radiance.rgb, variance, albedo.rgb, and density into sourceTensor
        // Channels: 0-2: Radiance, 3: Variance, 4-6: Albedo, 7: Density
        _inputCompilerShader.RunKernel(_inputCompilerKernel, simulation.width, simulation.height,
            ("radiance", simulation.SimulationOutput),
            ("variance", simulation.VarianceMap),
            ("albedo", simulation.GBuffer.AlbedoAlpha),
            ("transmissibility", simulation.GBuffer.Transmissibility),
            ("output", sourceTensor));

        // Push input tensor through model
        aiWorker.Schedule(sourceTensor);
        //outputTensor = aiWorker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

        // Push output tensor to final texture
        //Unity.InferenceEngine.TextureConverter.RenderToTexture(outputTensor, HDROutputTexture);

        sourceTensor.Dispose();
        //outputTensor.Dispose();
    }
}