using UnityEngine;

public class HybridTracer : Disposable, ITracer
{
    private ForwardMonteCarlo _forwardIntegrator;
    private BackwardMonteCarlo _backwardIntegrator;

    public PhotonerGBuffer GBuffer
    {
        get => _forwardIntegrator.GBuffer;
        set
        {
            _forwardIntegrator.GBuffer = value;
            _backwardIntegrator.GBuffer = value;
            _backwardIntegrator.InputImage = _forwardIntegrator.OutputImageHDR;
        }
    }

    public RenderTexture EarlyRadianceForImportanceSampling => _forwardIntegrator.OutputImageHDR;
    public RenderTexture TracerOutput => _backwardIntegrator.OutputImage;

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

    public float ForwardIntegrationInterval
    {
        get => _forwardIntegrator.IntegrationInterval;
        set => _forwardIntegrator.IntegrationInterval = value;
    }

    public float BackwardIntegrationInterval
    {
        get => _backwardIntegrator.IntegrationInterval;
        set => _backwardIntegrator.IntegrationInterval = value;
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

    public HybridTracer()
    {
        _forwardIntegrator = new ForwardMonteCarlo();
        AutoDispose(_forwardIntegrator);
        _backwardIntegrator = new BackwardMonteCarlo();
        AutoDispose(_backwardIntegrator);
    }

    // TODO Somewhere in here belongs importance sampling logic.

    public void NewScene()
    {
        _forwardIntegrator.IterationsSinceClear = 0;
        _forwardIntegrator.Clear();
        _backwardIntegrator.Clear();
    }

    public void BeginTrace(params RTLightSource[] lights)
    {
        _forwardIntegrator.IterationsSinceClear++;
        _forwardIntegrator.ImportanceSamplingTarget = new Vector2(0.5f, 0.5f);
        _forwardIntegrator.Integrate(lights);
    }

    public void EndTrace(RenderTexture importanceMap)
    {
        _backwardIntegrator.ImportanceMap = importanceMap;
        _backwardIntegrator.ImportanceSamplingTarget = new Vector2(0.5f, 0.5f);
        _backwardIntegrator.Integrate();
    }
}