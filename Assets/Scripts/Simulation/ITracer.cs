using System;
using UnityEngine;

public interface ITracer : IDisposable
{
    LitboxGBuffer GBuffer { get; set; }
    RenderTexture EarlyRadianceForImportanceSampling { get; }
    RenderTexture TracerOutput { get; }
    Matrix4x4 WorldToTargetTransform { get; set; }
    bool SkipAccumulation { get; set; }

    long ForwardWritesPerSecond { get; }
    long BackwardReadsPerSecond { get; }

    void NewScene();

    // Run tracing as much as required to generate a radiance map usable for importance sampling.
    void BeginTrace(params RTLightSource[] lights);

    // Finish tracing, optionally with an importance map for importance sampling
    void EndTrace(RenderTexture importanceMap = null);

    void UpdatePerformanceMetrics();
}

public interface ITracerDebug
{
    RenderTexture ForwardRawPhotons { get; }
    RenderTexture ForwardAccumulation { get; }
}