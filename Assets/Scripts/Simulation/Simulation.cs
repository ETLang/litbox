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
    public float integrationInterval;
    public int photonBounces;
}

public enum SimulationMode
{
    Realtime,
    Reference
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

    [SerializeField] public SimulationMode priority = SimulationMode.Realtime;
    [SerializeField] private int raysPerFrame = 64000;
    [SerializeField] private int photonBounces = -1;
    [SerializeField, Min(0.01f)] private float integrationInterval = 0.1f;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;

    [Header("Convergence Information")]
    [SerializeField, Min(1)] int _measurementInterval = 100;
    [SerializeField] private float _convergenceThreshold = -1;
    [SerializeField] private int iterationsSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    [Header("Archaic Properties")]
    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private bool bilinearPhotonWrites = true;

    private static int _MainTexID = Shader.PropertyToID("_MainTex");
    private Material _compositorMat;
    private SimulationCamera _realContentCamera;
    private bool _freshContentCamera = true;

    [SerializeField, ReadOnly] public bool hasConverged = false;

    private SortedDictionary<float, uint> performanceCounter = new SortedDictionary<float, uint>();

    public uint TraversalsPerSecond { get; private set; }
    public uint PhotonWritesPerSecond { get; private set; }

    int _sceneId;
    bool _validationFailed = false;
    bool _dirtyFrame;
    RTLightSource[] _allLights;
    RTObject[] _allObjects;
    Matrix4x4 _worldToPresentation;
    MeshFilter _meshFilter;
    MeshRenderer _meshRenderer;
    ITracer[] _activeTracer = new ITracer[2]; // Two parallel tracers allow easy computation of variance
    ConvergenceMeasurement _convergenceMeasurement;
    ImportanceMap _importanceMap;
    Action _updateTracerProperties;

    public PhotonerGBuffer GBuffer { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
    public RenderTexture VarianceMap { get; private set; }
    public RenderTexture ImportanceMap => _importanceMap?.Map;

    public ITracer TracerA => _activeTracer[0];
    public ITracer TracerB => _activeTracer[1];

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
        integrationInterval = profile.integrationInterval;
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

    void TracerTask(Action<ITracer> task)
    {
        for(int i = 0;i < 2;i++)
        {
            if(_activeTracer[i] != null) {
                task(_activeTracer[i]);
            }
        }
    }

    void ValidateTracer()
    {
        if(strategy == Strategy.LightTransport && !(_activeTracer[0] is LightTransportTracer))
        {
            var lightTracers = new LightTransportTracer[2];

            for(int i = 0;i < 2;i++) {
                _activeTracer[i]?.Dispose();
                _activeTracer[i] = lightTracers[i] = new LightTransportTracer();
            }

            _updateTracerProperties = () => {
                for(int i = 0;i < 2;i++)
                {
                    lightTracers[i].DisableBilinearWrites = !bilinearPhotonWrites;
                    lightTracers[i].IntegrationInterval = integrationInterval;
                    lightTracers[i].OverrideBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
                    lightTracers[i].RaysToEmit = raysPerFrame;
                }
            };
        } else if(strategy == Strategy.Hybrid && !(_activeTracer[0] is HybridTracer))
        {
            var hybridTracers = new HybridTracer[2];

            for(int i = 0;i < 2;i++) {
                _activeTracer[i]?.Dispose();
                _activeTracer[i] = hybridTracers[i] = new HybridTracer();
            }

            _updateTracerProperties = () =>
            {
                for(int i = 0;i < 2;i++) {
                    hybridTracers[i].DisableBilinearWrites = !bilinearPhotonWrites;
                    hybridTracers[i].ForwardIntegrationInterval = integrationInterval;
                    hybridTracers[i].BackwardIntegrationInterval = integrationInterval;
                    hybridTracers[i].OverrideBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
                    hybridTracers[i].RaysToEmit = raysPerFrame;
                }
            };
        }

        TracerTask(t => t.GBuffer = GBuffer);
    }

    private void Awake()
    {
#if UNITY_EDITOR
        _notPlaying = false;
#endif
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
        for(int i = 0;i < 2;i++) {
            _activeTracer[i]?.Dispose();
            _activeTracer[i] = null;
        }
        
#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged -= EditorApplication_playModeStateChanged;
#endif
    }

#if UNITY_EDITOR
    private bool _notPlaying;
    private void EditorApplication_playModeStateChanged(UnityEditor.PlayModeStateChange state)
    {
        if(state == UnityEditor.PlayModeStateChange.EnteredPlayMode && _notPlaying) {
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
        _convergenceMeasurement = new ConvergenceMeasurement();
        DisposeOnDisable(_convergenceMeasurement);

        _importanceMap = new ImportanceMap();
        DisposeOnDisable(_importanceMap);

        _realContentCamera = new GameObject("__Simulation_Camera", typeof(SimulationCamera)).GetComponent<SimulationCamera>();
        _realContentCamera.Initialize(transform, rayTracedLayers.value, priority);

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

            // Important to "prime" everything related to the camera
            _realContentCamera.GetComponent<Camera>().Render();
        }

        TracerTask(t => t.GBuffer = GBuffer);

        SimulationOutputHDR = this.CreateRWTextureWithMips(GBuffer.AlbedoAlpha.width, GBuffer.AlbedoAlpha.height, RenderTextureFormat.ARGBFloat);
        VarianceMap = this.CreateRWTexture(GBuffer.AlbedoAlpha.width / 4, GBuffer.AlbedoAlpha.height / 4, RenderTextureFormat.RFloat);
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

    bool _needsToUpdate = false;

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

        bool dirtyGBuffer = false;
        if (_dirtyFrame || _validationFailed)
        {
            hasConverged = false;
            _dirtyFrame = false;
            _validationFailed = false;
            dirtyGBuffer = true;
            iterationsSinceClear = 0;
            _sceneId++;
        }

        _needsToUpdate = true;
        if (frameLimit != -1)
        {
            if (iterationsSinceClear >= frameLimit) { _needsToUpdate = false; }
            hasConverged = false;
        }
        else if (hasConverged) { _needsToUpdate = false; }

        if(priority == SimulationMode.Realtime) {
            _realContentCamera.gameObject.SetActive(dirtyGBuffer);
        } else if(_needsToUpdate) {
            _realContentCamera.Render();
        }

        base.Update();
    }

    bool ShouldUpdateImportanceMap()
    {
        if(iterationsSinceClear == 0) { return priority != SimulationMode.Realtime; }
        else if(iterationsSinceClear < 100) { return iterationsSinceClear % 10 == 0; }
        else { return iterationsSinceClear % 100 == 0; }
    }

    void LateUpdate()
    {
#if UNITY_EDITOR
        if(_notPlaying) { return; }
#endif

        if (_realContentCamera == null) { return; }
        if (!GBuffer.IsValid) { return; }

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

        if(!_needsToUpdate) { return; }

        // CLEAR TARGET
        if (iterationsSinceClear == 0)
        {
            convergenceProgress = -1;
            ConvergenceStartTime = now;

            TracerTask(t => t?.NewScene());
        }

        iterationsSinceClear++;

        _updateTracerProperties();

        TracerTask(t =>
        {
            t.WorldToTargetTransform = worldToTargetSpace;
            t.BeginTrace(_allLights);
        });

        if (_activeTracer[0].EarlyRadianceForImportanceSampling != null && 
            _activeTracer[1].EarlyRadianceForImportanceSampling != null &&
            ShouldUpdateImportanceMap())
        {
            _importanceMap.Generate(
                _activeTracer[0].EarlyRadianceForImportanceSampling,
                _activeTracer[1].EarlyRadianceForImportanceSampling);
        }

        TracerTask(t => t.EndTrace(_importanceMap.Map));

        TracerPostProcessor.Instance.ComputeCVAndMips(_activeTracer[0].TracerOutput, _activeTracer[1].TracerOutput, SimulationOutputHDR, VarianceMap);

        _compositorMat.SetTexture(_MainTexID, SimulationOutputHDR);
        OnStep?.Invoke(iterationsSinceClear);

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

        if (_measurementInterval != 0 && iterationsSinceClear % _measurementInterval == 0 ||
            iterationsSinceClear == 1 && _convergenceThreshold > 0)
        {
           MeasureConvergence(iterationsSinceClear == 1);
        }

        if (fireConvergedEvent)
        {
            OnConverged?.Invoke();
        }
    }

    async void MeasureConvergence(bool initial)
    {
#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying) return;
#endif
        if (hasConverged) return;

        int recentSceneId = _sceneId;
        var variance = await _convergenceMeasurement.GetVarianceAsync(VarianceMap);

        if (recentSceneId != _sceneId) return;

#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying) return;
#endif

        convergenceProgress = variance;
        OnConvergenceUpdate?.Invoke(convergenceProgress);

        if (!initial && convergenceProgress < _convergenceThreshold)
        {
            hasConverged = true;
            OnConverged?.Invoke();
        }
    }
}