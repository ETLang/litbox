using UnityEngine;

public class LightTransportTracer : Disposable, ITracer
{
    private ForwardMonteCarlo _forwardIntegrator;

    public RenderTexture TracerOutput => _forwardIntegrator.OutputImageHDR;

    public PhotonerGBuffer GBuffer
    {
        get => _forwardIntegrator.GBuffer;
        set => _forwardIntegrator.GBuffer = value;
    }

    public bool DisableBilinearWrites
    {
        get => _forwardIntegrator.DisableBilinearWrites;
        set => _forwardIntegrator.DisableBilinearWrites = value;
    }

    public Matrix4x4 WorldToTargetTransform
    {
        get => _forwardIntegrator.WorldToTargetTransform;
        set => _forwardIntegrator.WorldToTargetTransform = value;
    }

    public float IntegrationInterval
    {
        get => _forwardIntegrator.IntegrationInterval;
        set => _forwardIntegrator.IntegrationInterval = value;
    }

    public uint? OverrideBounceCount
    {
        get => _forwardIntegrator.OverrideBounceCount;
        set => _forwardIntegrator.OverrideBounceCount = value;
    }

    public int RaysToEmit
    {
        get => _forwardIntegrator.RaysToEmit;
        set => _forwardIntegrator.RaysToEmit = value;
    }

    public LightTransportTracer()
    {
        _forwardIntegrator = new ForwardMonteCarlo();
        AutoDispose(_forwardIntegrator);
    }

    // TODO Somewhere in here belongs importance sampling logic.

    public void NewScene()
    {
        _forwardIntegrator.IterationsSinceClear = 0;
        _forwardIntegrator.Clear();
    }

    public void Trace(params RTLightSource[] lights)
    {
        _forwardIntegrator.IterationsSinceClear++;
        _forwardIntegrator.Integrate(lights);
    }
}