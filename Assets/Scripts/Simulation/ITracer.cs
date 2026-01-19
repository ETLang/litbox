using System;
using UnityEngine;

public interface ITracer : IDisposable
{
    PhotonerGBuffer GBuffer { get; set; }
    RenderTexture TracerOutput { get; }
    Matrix4x4 WorldToTargetTransform { get; set; }

    void NewScene();
    void Trace(params RTLightSource[] lights);
}