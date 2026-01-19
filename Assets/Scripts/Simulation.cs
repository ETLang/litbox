using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


public delegate void SimulationStepEvent(int frameCount);
public delegate void SimulationConvergedEvent();

[Serializable]
public struct SimulationProfile {
    public int frameLimit;
    public int raysPerFrame;
    public int photonBounces;
}

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class Simulation : PhotonerComponent
{
    public enum Strategy
    {
        LightTransport,
        Hybrid
    }

    [SerializeField] private LayerMask rayTracedLayers;

    [SerializeField] private int frameLimit = -1;
    [SerializeField] public int width = 256;
    [SerializeField] public int height = 256;

    [SerializeField] private int raysPerFrame = 64000;
    [SerializeField] private int photonBounces = -1;
    [SerializeField] private float integrationInterval = 0.1f;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;

    [Header("Convergence Information")]
    [SerializeField] private float _convergenceThreshold = -1;
    [SerializeField] private int iterationsSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    [Header("Archaic Properties")]
    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private bool bilinearPhotonWrites = true;

    private static int _MainTexID = Shader.PropertyToID("_MainTex");
    private Material _compositorMat;
    private SimulationCamera _realContentCamera;
    private ComputeShader _computeShader;

    private int _currentRenderTextureIndex = 0;
    private bool awaitingConvergenceResult = false;
    [SerializeField, ReadOnly] public bool hasConverged = false;

    private SortedDictionary<float, uint> performanceCounter = new SortedDictionary<float, uint>();

    public uint TraversalsPerSecond { get; private set; }
    public uint PhotonWritesPerSecond { get; private set; }
    private ulong _previousCumulativePhotons;
    private float _previousConvergenceFeedbackTime;

    int _sceneId;
    bool _validationFailed = false;
    bool _dirtyFrame;
    RTLightSource[] _allLights;
    RTObject[] _allObjects;
    Matrix4x4 _worldToPresentation;
    MeshFilter _meshFilter;
    MeshRenderer _meshRenderer;
    ITracer _activeTracer;

    public PhotonerGBuffer GBuffer { get; private set; }
    public RenderTexture SimulationOutputHDR => _activeTracer?.TracerOutput;
    public Texture2D EfficiencyDiagnostic { get; private set; }
    public Texture2D CumulativePhotonsDiagnostic { get; private set; }
    public Texture2D PhotonsDiagnostic { get; private set; }
    public Texture2D MaxValueDiagnostic { get; private set; }

    public float[,] EfficiencyData { get; private set; }
    public Vector2[,] EfficiencyGradient { get; private set; }
    public Vector2[,] RelaxedEfficiencyGradient { get; private set; }

    public Vector2 ImportanceSamplingTarget { get; private set; } = new Vector2(0.5f, 0.5f);
    public float ConvergenceStartTime { get; private set; }
    public float Convergence => convergenceProgress;
    public float EstimatedConvergenceTime => (Time.time - ConvergenceStartTime) * Convergence / _convergenceThreshold;

    public event SimulationStepEvent OnStep;
    public event SimulationConvergedEvent OnConverged;
    public event Action<float> OnConvergenceUpdate;

    private static Mesh _quadMesh;

    public static Mesh GetBuiltInQuadMesh()
    {
        if (_quadMesh == null) {
            GameObject tempQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _quadMesh = tempQuad.GetComponent<MeshFilter>().sharedMesh;
            GameObject.DestroyImmediate(tempQuad);
        }
        return _quadMesh;
    }

    public Simulation()
    {
        DetectSetChanges(() => _allLights, "dirtyFrame");
        DetectSetChanges(() => _allObjects, "dirtyFrame");
        DetectChanges(() => _worldToPresentation, "dirtyFrame");
        DetectChanges(() => integrationInterval, "dirtyFrame");
        DetectChanges(() => strategy, "dirtyFrame");
    }

    protected override void OnInvalidated(string group) => _dirtyFrame |= group == "dirtyFrame";

    public void LoadProfile(SimulationProfile profile)
    {
        frameLimit = profile.frameLimit;
        raysPerFrame = profile.raysPerFrame;
        photonBounces = profile.photonBounces;
        hasConverged = false;
        iterationsSinceClear = 0;
    }

    void ValidateTargets()
    {
        if (!GBuffer.IsValid || GBuffer.AlbedoAlpha.width != width || GBuffer.AlbedoAlpha.height != height)
        {
            CreateTargetBuffers();
            _validationFailed = true;
        }
    }

    void ValidateTracer()
    {
        var lightTransport = _activeTracer as LightTransportTracer;
        var hybrid = _activeTracer as HybridTracer;

        if(strategy == Strategy.LightTransport && lightTransport == null)
        {
            _activeTracer?.Dispose();
            _activeTracer = lightTransport = new LightTransportTracer();
        } else if(strategy == Strategy.Hybrid && hybrid == null)
        {
            _activeTracer?.Dispose();
            _activeTracer = hybrid = new HybridTracer();
        }

        _activeTracer.GBuffer = GBuffer;
    }

    private void Awake()
    {
        _computeShader = (ComputeShader)Resources.Load("Simulation");

        OnStartedPlaying();

        _meshFilter = GetComponent<MeshFilter>();
        _meshRenderer = GetComponent<MeshRenderer>();

        if(_meshFilter.sharedMesh == null) {
            _meshFilter.sharedMesh = GetBuiltInQuadMesh();
        }

        if(_meshRenderer.sharedMaterial == null) {
            _compositorMat = new Material(Shader.Find("Photoner/SimulationCompositor"));
            _meshRenderer.sharedMaterial = _compositorMat;
        } else {
            _compositorMat = _meshRenderer.sharedMaterial;
        }

#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged += EditorApplication_playModeStateChanged;
#endif
    }

    protected override void OnDestroy()
    {
        _activeTracer?.Dispose();
        _activeTracer = null;
        
#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged -= EditorApplication_playModeStateChanged;
#endif
    }

#if UNITY_EDITOR
    private bool _notPlaying;
    private void EditorApplication_playModeStateChanged(UnityEditor.PlayModeStateChange state)
    {
        if(state == UnityEditor.PlayModeStateChange.EnteredPlayMode) {
            _notPlaying = false;
            OnStartedPlaying();
        }

        if(state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
        {
            _notPlaying = true;
        }
    }
#endif

    private void OnStartedPlaying()
    {
        _realContentCamera = new GameObject("__Simulation_Camera", typeof(SimulationCamera)).GetComponent<SimulationCamera>();
        _realContentCamera.Initialize(transform, rayTracedLayers.value);

        CreateTargetBuffers();
    }

    private void CreateTargetBuffers()
    {
        PhotonerGBuffer gBuffer = new PhotonerGBuffer
        {
            AlbedoAlpha = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat, 32),
            Transmissibility = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat),
            NormalRoughness = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat),
            QuadTreeLeaves = this.CreateRWTexture(width, height, RenderTextureFormat.ARGBHalf),
        };

        GBuffer = gBuffer;

        if(_realContentCamera != null) {
            _realContentCamera.GBuffer = GBuffer;
            _realContentCamera.VarianceEpsilon = transmissibilityVariationEpsilon;
        }

        if(_activeTracer != null) {
            _activeTracer.GBuffer = gBuffer;
        }
    }

    void OnEnable()
    {
        ValidateTracer();
        ValidateTargets();
    }

    protected override void OnDisable()
    {
        if (_realContentCamera != null) {
            _realContentCamera.ClearTargets();
            Destroy(_realContentCamera);
            _realContentCamera = null;
        }

        base.OnDisable();
    }

    protected override void Update()
    {
        ValidateTracer();
        ValidateTargets();

        if (_realContentCamera == null) { return; }

        _worldToPresentation = _realContentCamera.transform.worldToLocalMatrix;
        _allLights = FindObjectsByType<RTLightSource>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        _allObjects = FindObjectsByType<RTObject>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);

        if((_allLights != null && _allLights.Any(x => x.Changed)) ||
           (_allObjects != null && _allObjects.Any(x => x.Changed)))
        {
            OnInvalidated("dirtyFrame");
        }

        base.Update();
    }

    const int ConvergenceMeasurementInterval = 100;
    void LateUpdate()
    {
#if UNITY_EDITOR
        if(_notPlaying) { return; }
#endif

        if (_realContentCamera == null) { return; }
        if (!GBuffer.IsValid) { return; }

        _realContentCamera.Render();

        var presentationToTargetSpace = Matrix4x4.Scale(new Vector3(width, height, 1)) * Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        var worldToTargetSpace = presentationToTargetSpace * _worldToPresentation;

        // PERFORMANCE MEASUREMENT
        var now = Time.time;
        while (performanceCounter.Keys.Count != 0 && performanceCounter.Keys.First() < now - 1)
            performanceCounter.Remove(performanceCounter.Keys.First());

        uint total = 0;
        foreach (var value in performanceCounter.Values)
            total += value;
        uint bouncesThisFrame = 0;

        foreach (var bounces in _allLights.Select(light => light.bounces))
            bouncesThisFrame += bounces;

        bouncesThisFrame *= (uint)raysPerFrame;

        uint existing = 0;
        performanceCounter.TryGetValue(now, out existing);
        performanceCounter[now] = existing + bouncesThisFrame;

        TraversalsPerSecond = total;

        // CHANGE DETECTION
        if (_dirtyFrame || _validationFailed)
        {
            hasConverged = false;
            _dirtyFrame = false;
            _validationFailed = false;
            iterationsSinceClear = 0;
            _sceneId++;
        }

        if (frameLimit != -1)
        {
            if (iterationsSinceClear >= frameLimit) { return; }
            hasConverged = false;
        }
        else if (hasConverged) { return; }

        // CLEAR TARGET
        if (iterationsSinceClear == 0)
        {
            awaitingConvergenceResult = false;
            convergenceProgress = -1;
            ConvergenceStartTime = now;

            _activeTracer?.NewScene();
        }

        iterationsSinceClear++;

        if(_activeTracer is LightTransportTracer lightTracer)
        {
            lightTracer.DisableBilinearWrites = !bilinearPhotonWrites;
            lightTracer.IntegrationInterval = integrationInterval;
            lightTracer.OverrideBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
            lightTracer.RaysToEmit = raysPerFrame;
        }

        if(_activeTracer is HybridTracer hybridTracer)
        {
            hybridTracer.DisableBilinearWrites = !bilinearPhotonWrites;
            hybridTracer.ForwardIntegrationInterval = integrationInterval;
            hybridTracer.BackwardIntegrationInterval = integrationInterval;
            hybridTracer.OverrideBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
            hybridTracer.RaysToEmit = raysPerFrame;
        }

        _activeTracer.WorldToTargetTransform = worldToTargetSpace;
        _activeTracer.Trace(_allLights);

        // Generate mip levels (todo: this can be much faster)
        int mipW = SimulationOutputHDR.width;
        int mipH = SimulationOutputHDR.height;
        for(int i = 1;i < SimulationOutputHDR.mipmapCount;i++) {
            mipW /= 2;
            mipH /= 2;
            _computeShader.RunKernel("GenerateOutputMips", mipW, mipH,
                ("g_sourceMipLevelHDR", SimulationOutputHDR.SelectMip(i - 1)),
                ("g_destMipLevelHDR", SimulationOutputHDR.SelectMip(i)));
        }

        _compositorMat.SetTexture(_MainTexID, SimulationOutputHDR);
        OnStep?.Invoke(iterationsSinceClear);

        // CONVERGENCE TESTING (todo: Overhaul entirely)
        bool fireConvergedEvent = false;
        if (frameLimit != -1 && iterationsSinceClear >= frameLimit)
        {
            if (iterationsSinceClear > frameLimit)
            {
                Debug.LogError("Skipped a frame somehow...");
            }

            hasConverged = true;
            fireConvergedEvent = true;
        }

        if (ConvergenceMeasurementInterval != 0 && iterationsSinceClear % ConvergenceMeasurementInterval == 0 ||
            iterationsSinceClear == 1 && _convergenceThreshold > 0)
        {
           // MeasureConvergence(iterationsSinceClear == 1);
        }

        if (fireConvergedEvent)
        {
            OnConverged?.Invoke();
        }
    }

    //async void MeasureConvergence(bool initial)
    void MeasureConvergence(bool initial)
    {
        // Convergence: The change in output image approaches zero.
        // There's two ways to measure change:
        //    A. Pixel difference per frame
        //    B. Pixel difference per photon
        // The challenge with A is that some gridcells receive very few photons per frame.
        // This means their delta per frame is very small, but only because there's nowhere near
        // enough information to tell how it's converging.
        //
        // The challenge with B is that some situations aren't supposed to receive very many photons,
        // so you can't rely on simply firing a lot of photons into a space and expecting many results.
        //
        // Different scenarios to consider:
        //  1. Low density cells (little photon interaction expected)
        //  2. Obscured cells (little photon interaction without targeted sampling)
        //  3. High variance cells
        //
        // Expected interaction rate can be read from G buffer.
        // This allows us to measure how well a cell is being sampled.
        // Sample Efficiency E = P / ((1 - T) * A)
        // T: Transmissibility
        // P: Photon Count
        // A: Cell Area (# of pixels)
        //
        // if E is large enough, then we can assume that the cell is converging.
        // If E is too small, it is not converging and it doesn't matter if the change rate is small.
        // If E is large and Pixel delta is small, we're converging!
        // 

#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying) return;
#endif
        if (awaitingConvergenceResult) return;
        awaitingConvergenceResult = true;
        if (hasConverged) return;

        // TODO: Measure convergence via variance map
        // _computeShader.RunKernel("MeasureConvergence", width, height,
        //     ("g_output_raw", SimulationPhotonsRaw),
        //     ("g_output_tonemapped", _renderTexture[_currentRenderTextureIndex]),
        //     ("g_previousResult", _renderTexture[1 - _currentRenderTextureIndex]),
        //     ("g_convergenceCellStateIn", _gridCellInputBuffer),
        //     ("g_convergenceCellStateOut", _gridCellOutputBuffer));
        _currentRenderTextureIndex = 1 - _currentRenderTextureIndex;

        int recentSceneId = _sceneId;
       // var r = await AsyncGPUReadback.RequestAsync(_gridCellOutputBuffer);

        if (recentSceneId != _sceneId) return;
        awaitingConvergenceResult = false;
        //if (!r.done || r.hasError) return;

#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying) return;
#endif

        //convergenceProgress = overallConvergence / cellArea;
        OnConvergenceUpdate?.Invoke(convergenceProgress);

        if (!initial && convergenceProgress < _convergenceThreshold)
        {
            hasConverged = true;
            OnConverged?.Invoke();
        }
    }
}