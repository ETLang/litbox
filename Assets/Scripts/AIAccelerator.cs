#define DISABLE

using System;
using UnityEngine;


public class AIAccelerator : MonoBehaviour {
    [SerializeField] private Simulation simulation;
    [SerializeField] private bool operateOnToneMapped;

    public RenderTexture HDROutputTexture { get; private set; }
    public RenderTexture ToneMappedOutputTexture { get; private set; }

#if !DISABLE
    [SerializeField] private Unity.InferenceEngine.ModelAsset accelerationModel;

    Unity.InferenceEngine.Worker aiWorker;
    Unity.InferenceEngine.Tensor<float> sourceTensor;


    void Start() {
        if(simulation) {
            simulation.OnStep += Simulation_OnStep;
            HDROutputTexture = new RenderTexture(simulation.TextureResolution, simulation.TextureResolution, 0, RenderTextureFormat.ARGBFloat);
            HDROutputTexture.Create();
            ToneMappedOutputTexture = new RenderTexture(simulation.TextureResolution, simulation.TextureResolution, 0, RenderTextureFormat.ARGB32);
            ToneMappedOutputTexture.Create();
        }

        var model = Unity.InferenceEngine.ModelLoader.Load(accelerationModel);
        aiWorker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.GPUCompute);

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
        Unity.InferenceEngine.Tensor<float> outputTensor = null;

        if(operateOnToneMapped) {
            // Push output texture to input tensor
            sourceTensor = Unity.InferenceEngine.TextureConverter.ToTensor(simulation.SimulationOutputToneMapped);
            
            // Push input tensor through model
            aiWorker.Schedule(sourceTensor);
            outputTensor = aiWorker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

            // Push output tensor to final texture
            Unity.InferenceEngine.TextureConverter.RenderToTexture(outputTensor, ToneMappedOutputTexture);
        } else {
            // Push output texture to input tensor
            sourceTensor = Unity.InferenceEngine.TextureConverter.ToTensor(simulation.SimulationOutputHDR);
            
            // Push input tensor through model
            aiWorker.Schedule(sourceTensor);
            outputTensor = aiWorker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

            // Push output tensor to final texture
            Unity.InferenceEngine.TextureConverter.RenderToTexture(outputTensor, HDROutputTexture);
        }

        sourceTensor.Dispose();
        outputTensor.Dispose();
    }
#endif
}