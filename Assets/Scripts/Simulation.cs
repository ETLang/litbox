using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Linq;
using System.Threading;


public delegate void SimulationStepEvent(int frameCount);
public delegate void SimulationConvergedEvent();

[Serializable]
public struct SimulationProfile {
    public int frameLimit;
    public int resolution;
    public int threadCount;
    public int photonsPerThread;
    public int photonBounces;
    public int energyUnit;
    public float transmissibilityVariationEps;
    public float outscatterCoefficient;
}

public class Simulation : SimulationBaseBehavior
{
    public enum Strategy {
        LightTransport,
        PathTracing,
        Hybrid
    }

    [SerializeField] private LayerMask rayTracedLayers;

    [SerializeField] private int frameLimit = -1;
    [SerializeField] private int textureResolution = 256;

    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private int threadCount = 4096;
    [SerializeField] private int photonsPerThread = 4096;
    [SerializeField] private int photonBounces = -1;
    [SerializeField] private int pathSamples = 10;
    [SerializeField, Range(0,1)] private float pathBalance = 0.5f;

    [SerializeField] private int energyUnit = 100000;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;
    [SerializeField, Range(0,0.5f)] private float outscatterCoefficient = 0.01f;

    private SimulationCamera _realContentCamera;
    private ComputeShader _computeShader;
    private ComputeBuffer _randomBuffer;
    private ComputeBuffer _measureConvergenceResultBuffer;

    private RenderTexture[] _renderTexture = new RenderTexture[2];
    private int _currentRenderTextureIndex = 0;
    private RenderBuffer[][] _gBuffer;
    private Texture _mieScatteringLUT;
    private Texture _teardropScatteringLUT;
    private int[] _kernelsHandles;

    private Renderer _renderer;

    [Header("Convergence Information")]
    [SerializeField] private float _convergenceThreshold = 10;
    [SerializeField] private int framesSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    private bool awaitingConvergenceResult = false;
    [SerializeField, ReadOnly] public bool hasConverged = false;

    private uint4[] convergenceResultResetData = new uint4[] { new uint4(0, 0, 0, 0) };
    private SortedDictionary<float,uint> performanceCounter = new SortedDictionary<float, uint>();

    public uint TraversalsPerSecond { get; private set; }

    private RenderTexture[] _gBufferAlbedo = new RenderTexture[2];
    private RenderTexture[] _gBufferTransmissibility = new RenderTexture[2];
    private RenderTexture[] _gBufferNormalSlope = new RenderTexture[2];
    private int _gBufferNextTarget = 0;

    public RenderTexture GBufferAlbedo { get; private set; }
    public RenderTexture GBufferTransmissibility { get; private set; }
    public RenderTexture GBufferNormalSlope { get; private set; }
    public RenderTexture GBufferQuadTreeLeaves { get; private set; }

    public RenderTexture SimulationForwardOfHybrid { get; private set; }
    public RenderTexture SimulationOutputRaw { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
    public RenderTexture SimulationOutputToneMapped { get; private set; }

    public float ConvergenceStartTime { get; private set; }
    public float Convergence => convergenceProgress;
    public int TextureResolution => textureResolution;

    public event SimulationStepEvent OnStep;
    public event SimulationConvergedEvent OnConverged;

    public void LoadProfile(SimulationProfile profile) {
        frameLimit = profile.frameLimit;
        textureResolution = profile.resolution;
        threadCount = profile.threadCount;
        photonsPerThread = profile.photonsPerThread;
        energyUnit = profile.energyUnit;
        transmissibilityVariationEpsilon = profile.transmissibilityVariationEps;
        outscatterCoefficient = profile.outscatterCoefficient;
        photonBounces = profile.photonBounces;
        hasConverged = false;
        framesSinceClear = 0;
    }

    private void Start()
    {
        _computeShader = (ComputeShader)Resources.Load("Test_Compute");

        //GET RENDERER COMPONENT REFERENCE
        TryGetComponent(out _renderer);

        var randSeeds = Math.Max(threadCount, textureResolution * textureResolution);

        uint4[] seeds = new uint4[randSeeds];

        for(int i = 0;i < seeds.Length;i++) {
            seeds[i].x = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].y = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].z = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].w = (uint)(UnityEngine.Random.value * 1000000);
        }

        _randomBuffer = CreateStructuredBuffer(seeds);
        _measureConvergenceResultBuffer = CreateStructuredBuffer(convergenceResultResetData);

        for(int i = 0;i < _renderTexture.Length;i++) {
            _renderTexture[i] = CreateRWTexture(textureResolution, textureResolution, RenderTextureFormat.DefaultHDR);
        }

        SimulationForwardOfHybrid = CreateRWTexture(textureResolution * 3, textureResolution, RenderTextureFormat.RInt);
        SimulationOutputRaw = CreateRWTexture(textureResolution * 3, textureResolution, RenderTextureFormat.RInt);
        SimulationOutputHDR = CreateRWTexture(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);

        for(int i = 0;i < 2;i++) {
            _gBufferAlbedo[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
            _gBufferTransmissibility[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
            _gBufferNormalSlope[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
        }

        GBufferQuadTreeLeaves = CreateRWTexture(textureResolution, textureResolution, RenderTextureFormat.ARGBHalf);

        _mieScatteringLUT = LUT.CreateMieScatteringLUT();
        DisposeOnDisable(() => DestroyImmediate(_mieScatteringLUT));
        _teardropScatteringLUT = LUT.CreateTeardropScatteringLUT();
        DisposeOnDisable(() => DestroyImmediate(_teardropScatteringLUT));

        _realContentCamera = new GameObject("__Simulation_Camera", typeof(SimulationCamera)).GetComponent<SimulationCamera>();
        _realContentCamera.Initialize(transform, rayTracedLayers.value);

        SwapGBuffer();
    }

    private void SwapGBuffer() {
        GBufferAlbedo = _gBufferAlbedo[_gBufferNextTarget];
        GBufferTransmissibility = _gBufferTransmissibility[_gBufferNextTarget];
        GBufferNormalSlope = _gBufferNormalSlope[_gBufferNextTarget];

        //_gBufferNextTarget = 1 - _gBufferNextTarget;

        _realContentCamera.GBufferAlbedo = _gBufferAlbedo[_gBufferNextTarget];
        _realContentCamera.GBufferTransmissibility = _gBufferTransmissibility[_gBufferNextTarget];
        _realContentCamera.GBufferNormalSlope = _gBufferNormalSlope[_gBufferNextTarget];
        _realContentCamera.GBufferQuadTreeLeaves = GBufferQuadTreeLeaves;
        _realContentCamera.VarianceEpsilon = transmissibilityVariationEpsilon;
    }

    protected override void OnDisable()
    {
        _realContentCamera.ClearTargets();
        Destroy(_realContentCamera);
        _realContentCamera = null;

        base.OnDisable();
    }

    Matrix4x4 _previousSimulationMatrix;
    HashSet<RTLightSource> _previousLightSources = new HashSet<RTLightSource>();
    HashSet<RTObject> _previousObjects = new HashSet<RTObject>();
    float _previousOutscatterCoefficient;
    float _previousPathBalance;
    int _sceneId;
    bool CheckChanged(RTLightSource[] allLights, RTObject[] allObjects, Matrix4x4 worldToPresentation) {
        var changed = 
            allLights.Length != _previousLightSources.Count ||
            !allLights.All(l => _previousLightSources.Contains(l)) ||
            allLights.Any(l => l.Changed) ||
            allObjects.Length != _previousObjects.Count ||
            !allObjects.All(o => _previousObjects.Contains(o)) ||
            allObjects.Any(o => o.Changed) ||
            _previousSimulationMatrix != worldToPresentation ||
            _previousOutscatterCoefficient != outscatterCoefficient ||
            _previousPathBalance != pathBalance;

        _previousPathBalance = pathBalance;
        _previousOutscatterCoefficient = outscatterCoefficient;
        _previousSimulationMatrix = worldToPresentation;
        _previousLightSources.Clear();
        foreach(var light in allLights)
            _previousLightSources.Add(light);
        _previousObjects.Clear();
        foreach(var o in allObjects)
            _previousObjects.Add(o);

        return changed;
    }

    void Update() {
        _realContentCamera.Render();

        var worldToPresentation = transform.worldToLocalMatrix;
        var presentationToTargetSpace = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        var worldToTargetSpace = presentationToTargetSpace * worldToPresentation;
        double photonEnergy = (double)uint.MaxValue / threadCount;
        var allLights = FindObjectsByType<RTLightSource>(FindObjectsSortMode.None);
        var allObjects = FindObjectsByType<RTObject>(FindObjectsSortMode.None);

        // PERFORMANCE MEASUREMENT
        var now = Time.time;
        while(performanceCounter.Keys.Count != 0 && performanceCounter.Keys.First() < now - 1)
            performanceCounter.Remove(performanceCounter.Keys.First());
        
        uint total = 0;
        foreach(var value in performanceCounter.Values)
            total += value;
        uint bouncesThisFrame = 0;
        
        foreach(var bounces in allLights.Select(light => light.bounces))
            bouncesThisFrame += bounces;

        bouncesThisFrame *= (uint)photonsPerThread * (uint)threadCount;
        
        if(performanceCounter.TryGetValue(now, out var existing)) {
            performanceCounter[now] = existing + bouncesThisFrame;
        } else {
            performanceCounter[now] = bouncesThisFrame;
        }

        TraversalsPerSecond = total;

        // CHANGE DETECTION
        if(CheckChanged(allLights, allObjects, worldToPresentation)) {
            hasConverged = false;
            framesSinceClear = 0;
            _sceneId++;
        }

        if(frameLimit != -1) {
            if(framesSinceClear >= frameLimit)
                return;
            else
                hasConverged = false;
        } else if (hasConverged) {
            return;
        }

        // CLEAR TARGET
        if(framesSinceClear == 0) {
            awaitingConvergenceResult = false;
            convergenceProgress = -1;
            ConvergenceStartTime = now;

            SimulationOutputRaw.Clear(Color.clear);
        }

        framesSinceClear++;

        // RAY TRACING SIMULATION
        _computeShader.SetVector("g_target_size", new Vector2(textureResolution, textureResolution));
        _computeShader.SetInt("g_time_ms", Time.frameCount);
        _computeShader.SetInt("g_photons_per_thread", photonsPerThread);
        _computeShader.SetInt("g_samples_per_pixel", pathSamples);
        _computeShader.SetMatrix("g_worldToTarget", Matrix4x4.identity);
        _computeShader.SetFloat("g_TransmissibilityVariationEpsilon", transmissibilityVariationEpsilon);
        _computeShader.SetInt("g_lowest_lod", (int)(GBufferTransmissibility.mipmapCount - 4));
        _computeShader.SetInt("g_4x4_lod", (int)(GBufferTransmissibility.mipmapCount - 3));
        _computeShader.SetFloat("g_lightEmissionOutscatter", 0);
        _computeShader.SetFloat("g_outscatterCoefficient", outscatterCoefficient);

        float energyNormPerFrame = 1;
        float pixelCount = textureResolution * textureResolution;
        
        switch(strategy) {
            case Strategy.LightTransport:
                // Forward light tracing technique:
                // Photons are simulated from each light source.

                // At each collision point, a portion of the light's energy is "reflected" out to the viewer's eye (outscattered).
                energyNormPerFrame = photonsPerThread * threadCount / pixelCount;
                _computeShader.SetFloat("g_energy_norm", framesSinceClear * energyNormPerFrame * energyUnit);

                foreach(var light in allLights) {
                    SimulateLight(light, strategy, photonBounces != -1 ? photonBounces : (int)light.bounces, worldToTargetSpace, SimulationOutputRaw);
                }
                break;
            case Strategy.PathTracing:
                // Path tracing technique:
                // Paths are simulated starting at each pixel in the view.

                // First, light sources are rendered without propagation into the simulation field.

                // Then, paths are followed from each pixel in the view until they hit the max
                //     bounce count, escape the simulated area, or hit a light source.
                //     When a light source is hit, the value of the light is added to the associated pixel.

                // TODO
                SimulationForwardOfHybrid.Clear(Color.clear);
                break;
            case Strategy.Hybrid:
                // Hybrid forward/backward tracing technique

                // Photons are simulated from each light source into the simulated field for N bounces.
                //    This process is the same as it is for the LightTransport strategy except the render
                //    target is an intermediate buffer.

                // Then, paths are traced backward from view pixels through M bounces.
                //    When paths intersect a pixel that has energy deposited from a previous step,
                //    that energy is propagated to the tracing pixel.

                energyNormPerFrame = photonsPerThread * threadCount / pixelCount;
                _computeShader.SetFloat("g_energy_norm", framesSinceClear * energyNormPerFrame * energyUnit);
                _computeShader.SetFloat("g_path_balance", pathBalance);

                // Clear intermediate target
                SimulationForwardOfHybrid.Clear(Color.clear);

                foreach(var light in allLights) {
                    SimulateLight(light, Strategy.LightTransport, photonBounces != -1 ? photonBounces / 2 : (int)light.bounces, worldToTargetSpace, SimulationForwardOfHybrid);
                }

                RunKernel(_computeShader, "Simulate_View_Backward", textureResolution, textureResolution,
                    ("g_rand", _randomBuffer),
                    ("g_photons_forward", SimulationForwardOfHybrid),
                    ("g_photons_final", SimulationOutputRaw),
                    ("g_albedo", GBufferAlbedo),
                    ("g_transmissibility", GBufferTransmissibility),
                    ("g_normalSlope", GBufferNormalSlope),
                    ("g_quadTreeLeaves", GBufferQuadTreeLeaves),
                    ("g_mieScatteringLUT", _mieScatteringLUT),
                    ("g_teardropScatteringLUT", _teardropScatteringLUT));
                break;
        }

        // HDR MAPPING
        RunKernel(_computeShader, "ConvertToHDR", textureResolution, textureResolution,
            ("g_photons_final", SimulationOutputRaw),
            ("g_hdrResult", SimulationOutputHDR));

        // TONE MAPPING
        RunKernel(_computeShader, "ToneMap", textureResolution, textureResolution,
            ("g_photons_final", SimulationOutputRaw),
            ("g_result", _renderTexture[_currentRenderTextureIndex]));

        SimulationOutputToneMapped = _renderTexture[_currentRenderTextureIndex];

        OnStep?.Invoke(framesSinceClear);
        
        // CONVERGENCE TESTING
        if(frameLimit != -1 && framesSinceClear >= frameLimit) {
            if(framesSinceClear > frameLimit) {
                Debug.LogError("Skipped a frame somehow...");
            }

            hasConverged = true;
            OnConverged?.Invoke();
        }

        int framesPerTest = 100;
        if(!awaitingConvergenceResult && framesPerTest != 0 && framesSinceClear % framesPerTest == 0) {
            awaitingConvergenceResult = true;

            _measureConvergenceResultBuffer.SetData(convergenceResultResetData);

            RunKernel(_computeShader, "MeasureConvergence", textureResolution, textureResolution,
                ("g_photons_final", SimulationOutputRaw),
                ("g_result", _renderTexture[_currentRenderTextureIndex]),
                ("g_previousResult", _renderTexture[1-_currentRenderTextureIndex]),
                ("g_convergenceResult", _measureConvergenceResultBuffer));
            _currentRenderTextureIndex = 1 - _currentRenderTextureIndex;

            int recentSceneId = _sceneId;
            AsyncGPUReadback.Request(_measureConvergenceResultBuffer, (r) =>
            {
                if(recentSceneId != _sceneId) return;
                awaitingConvergenceResult = false;
                if(!r.done || r.hasError) return;

                var feedback = r.GetData<uint>(0);
                uint pixelChange = feedback[0];
                uint pixelsMaxed = feedback[1];

                convergenceProgress = pixelChange / 100.0f;
                if(hasConverged = pixelsMaxed > 0 || convergenceProgress < _convergenceThreshold) {
                    OnConverged?.Invoke();
                }
            });
        }

        SwapGBuffer();
    }

    void SimulateLight(RTLightSource light, Strategy strategy, int bounces, Matrix4x4 worldToTargetSpace, RenderTexture outputTexture) {
        int simulateKernel = -1;
        var lightToTargetSpace = worldToTargetSpace * light.WorldTransform;
        double photonEnergy = (double)uint.MaxValue / threadCount;

        string kernelFormat;

        switch(strategy) {
        case Strategy.LightTransport:
            kernelFormat = "Simulate_{0}";
            break;
        case Strategy.PathTracing:
            kernelFormat = "Simulate_{0}_Splat";
            break;
        case Strategy.Hybrid:
            kernelFormat = "Simulate_{0}_Forward";
            break;
        default:
            Debug.LogError("What?");
            return;
        }

        switch(light) {
        case RTPointLight pt:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "PointLight"));
            _computeShader.SetFloat("g_lightEmissionOutscatter", pt.emissionOutscatter);
            break;
        case RTSpotLight _:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "SpotLight"));
            break;
        case RTLaserLight _:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "LaserLight"));
            break;
        case RTAmbientLight _:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "AmbientLight"));
            break;
        case RTFieldLight field:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "FieldLight"));
            _computeShader.SetTexture(simulateKernel, "g_lightFieldTexture", field.lightTexture ? field.lightTexture : Texture2D.whiteTexture);
            _computeShader.SetFloat("g_lightEmissionOutscatter", field.emissionOutscatter);
            break;
        case RTDirectionalLight dir:
            simulateKernel = _computeShader.FindKernel(string.Format(kernelFormat, "DirectionalLight"));
            _computeShader.SetVector("g_directionalLightDirection", lightToTargetSpace.MultiplyVector(new Vector3(0,-1,0)));
            break;
        }


        _computeShader.SetVector("g_lightEnergy", light.Energy * (float)photonEnergy);
        _computeShader.SetInt("g_bounces", bounces);
        _computeShader.SetMatrix("g_lightToTarget", lightToTargetSpace.transpose);
        _computeShader.SetBuffer(simulateKernel, "g_rand", _randomBuffer);
        _computeShader.SetTexture(simulateKernel, "g_photons_final", outputTexture);
        _computeShader.SetTexture(simulateKernel, "g_albedo", GBufferAlbedo);
        _computeShader.SetTexture(simulateKernel, "g_transmissibility", GBufferTransmissibility);
        _computeShader.SetTexture(simulateKernel, "g_normalSlope", GBufferNormalSlope);
        _computeShader.SetTexture(simulateKernel, "g_quadTreeLeaves", GBufferQuadTreeLeaves);
        _computeShader.SetTexture(simulateKernel, "g_mieScatteringLUT", _mieScatteringLUT);
        _computeShader.SetTexture(simulateKernel, "g_teardropScatteringLUT", _teardropScatteringLUT);

        _computeShader.Dispatch(simulateKernel, threadCount / 64, 1, 1);
    }
}
