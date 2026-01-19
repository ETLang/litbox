using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System.Linq;
using Unity.VisualScripting;


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
    public enum IntegrationMethod
    {
        Explicit,
        Implicit,
        ImplicitInterval,
        ExplicitBounded,
        ExplicitBounceImplicitInterval,
        ExplicitBoundedBounceImplicitInterval
    }

    public enum Strategy
    {
        LightTransport,
        Hybrid
    }

    struct ConvergenceCellInput
    {
        public uint IsActive;
        public uint HasConverged;
        public uint FrameCount;
        public float TargetTransmission;
        public float TransmissionSpread;
        public float Reserved0;
        public float Reserved1;
        public float Reserved2;
    }

    struct ConvergenceCellOutput
    {
        public uint MaxValue;
        public uint PixelChange;
        public uint PhotonCount;
        public float Transmissibility;
    }

    struct ConvergenceCellGroup
    {
        public int StartX;
        public int StartY;
        public int EndX;
        public int EndY;
    }

    struct ConvergenceCellState
    {
        public uint CumulativePhotonCount;
    }

    [SerializeField] private LayerMask rayTracedLayers;

    [SerializeField] private int frameLimit = -1;
    [RenamedFrom("textureResolution")]
    [SerializeField] public int width = 256;
    [SerializeField] public int height = 256;

    [SerializeField] private int raysPerFrame = 64000;
    [SerializeField] private int photonBounces = -1;
    [SerializeField] private IntegrationMethod integrationMethod = IntegrationMethod.ExplicitBoundedBounceImplicitInterval;
    [SerializeField] private float integrationInterval = 0.1f;
    [SerializeField] private int densityGranularity = 10000;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;

    [Header("Convergence Information")]
    [SerializeField] private float _convergenceThreshold = -1;
    [SerializeField] private int framesSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    [Header("Archaic Properties")]
    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private uint2 gridCells = new uint2(8, 8);
    [SerializeField] private bool bilinearPhotonWrites = true;
    [SerializeField] private int pathSamples = 10;

    private static int _MainTexID = Shader.PropertyToID("_MainTex");
    private Material _compositorMat;
    private SimulationCamera _realContentCamera;
    private ComputeShader _computeShader;
    private ComputeBuffer _gridCellInputBuffer;
    private ComputeBuffer _gridCellOutputBuffer;
    private List<ConvergenceCellGroup> _gridCellInputGroups = new List<ConvergenceCellGroup>();
    private ConvergenceCellInput[] _gridCellInput;
    private ConvergenceCellInput[] _gridCellInputInitialValue;
    private ConvergenceCellOutput[] _gridCellOutput;
    private ConvergenceCellOutput[] _gridCellOutputInitialValue;
    private ConvergenceCellState[] _gridCellState;

    private int _currentRenderTextureIndex = 0;
    private bool hasValidGridTransmissibility = false;
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
    ForwardMonteCarlo _forwardIntegrator;
    BackwardMonteCarlo _backwardIntegrator;

    public PhotonerGBuffer GBuffer { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
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
        DetectChanges(() => integrationMethod, "dirtyFrame");
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
        framesSinceClear = 0;
    }

    void ValidateTargets()
    {
        if (SimulationOutputHDR == null || SimulationOutputHDR.width != width || SimulationOutputHDR.height != height)
        {
            CreateTargetBuffers();
            _validationFailed = true;
        }
    }

    private void Awake()
    {
        _computeShader = (ComputeShader)Resources.Load("Simulation");
        _forwardIntegrator = new ForwardMonteCarlo();
        DisposeOnDisable(_forwardIntegrator);

        _backwardIntegrator = new BackwardMonteCarlo();
        DisposeOnDisable(_backwardIntegrator);

        OnStartedPlaying();

        if(strategy == Strategy.LightTransport)
        {
            SimulationOutputHDR = _forwardIntegrator.OutputImageHDR;
        } else
        {
            SimulationOutputHDR = _backwardIntegrator.OutputImage;
        }

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

        _gridCellState = new ConvergenceCellState[gridCells.x * gridCells.y];
        for (int i = 0; i < _gridCellState.Length; i++) {
            _gridCellState[i] = new ConvergenceCellState {
                CumulativePhotonCount = 0
            };
        }

        _gridCellInput = new ConvergenceCellInput[gridCells.x * gridCells.y];
        for (int i = 0; i < _gridCellInput.Length; i++) {
            _gridCellInput[i] = new ConvergenceCellInput {
                IsActive = 1,
                HasConverged = 0,
                FrameCount = 0,
                TargetTransmission = 1,
                TransmissionSpread = 0,
            };
        }
        _gridCellInputInitialValue = (ConvergenceCellInput[])_gridCellInput.Clone();
        _gridCellInputBuffer = this.CreateStructuredBuffer(_gridCellInput);

        _gridCellOutput = new ConvergenceCellOutput[gridCells.x * gridCells.y];
        for (int i = 0; i < _gridCellOutput.Length; i++) {
            _gridCellOutput[i] = new ConvergenceCellOutput {
                PixelChange = 0,
                PhotonCount = 0,
                MaxValue = 0
            };
        }
        _gridCellOutputInitialValue = (ConvergenceCellOutput[])_gridCellOutput.Clone();
        _gridCellOutputBuffer = this.CreateStructuredBuffer(_gridCellOutput);

        EfficiencyDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false) {
            filterMode = FilterMode.Point
        };
        PhotonsDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false) {
            filterMode = FilterMode.Point
        };
        MaxValueDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false) {
            filterMode = FilterMode.Point
        };

        EfficiencyData = new float[gridCells.x, gridCells.y];
        EfficiencyGradient = new Vector2[gridCells.x, gridCells.y];
        for (int i = 0; i < gridCells.x; i++) {
            for (int j = 0; j < gridCells.y; j++) {
                EfficiencyGradient[i, j] = new Vector2(0, 0);
            }
        }
        RelaxedEfficiencyGradient = new Vector2[gridCells.x, gridCells.y];
        for (int i = 0; i < gridCells.x; i++) {
            for (int j = 0; j < gridCells.y; j++) {
                RelaxedEfficiencyGradient[i, j] = new Vector2(0, 0);
            }
        }
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

        _forwardIntegrator.GBuffer = GBuffer;
        _backwardIntegrator.GBuffer = GBuffer;
        _backwardIntegrator.InputImage = _forwardIntegrator.OutputImageHDR;

        if(_realContentCamera != null) {
            _realContentCamera.GBuffer = GBuffer;
            _realContentCamera.VarianceEpsilon = transmissibilityVariationEpsilon;
        }
    }

    void OnEnable()
    {
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
        ValidateTargets();

        if (_realContentCamera == null)
        {
            return;
        }

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

        if (performanceCounter.TryGetValue(now, out var existing))
        {
            performanceCounter[now] = existing + bouncesThisFrame;
        }
        else
        {
            performanceCounter[now] = bouncesThisFrame;
        }

        TraversalsPerSecond = total;

        // CHANGE DETECTION
        if (_dirtyFrame || _validationFailed)
        {
            hasConverged = false;
            _dirtyFrame = false;
            _validationFailed = false;
            framesSinceClear = 0;
            _gridCellInput = (ConvergenceCellInput[])_gridCellInputInitialValue.Clone();
            _sceneId++;
        }

        if (frameLimit != -1)
        {
            if (framesSinceClear >= frameLimit)
                return;
            else
                hasConverged = false;
        }
        else if (hasConverged)
        {
            return;
        }

        // CLEAR TARGET
        if (framesSinceClear == 0)
        {
            hasValidGridTransmissibility = false;
            awaitingConvergenceResult = false;
            convergenceProgress = -1;
            ConvergenceStartTime = now;

            _forwardIntegrator.Clear();

            if(strategy == Strategy.Hybrid)
            {
                _backwardIntegrator.Clear();
            }

            for (int i = 0; i < _gridCellState.Length; i++)
            {
                _gridCellState[i].CumulativePhotonCount = 0;
            }

            for (int i = 0; i < _gridCellInput.Length; i++)
            {
                _gridCellInput[i].FrameCount = 0;
            }
            _gridCellInputBuffer.SetData(_gridCellInput);
        }

        framesSinceClear++;

        // RAY TRACING SIMULATION
        _computeShader.SetVector("g_importance_sampling_target", ImportanceSamplingTarget);
        _computeShader.SetVector("g_target_size", new Vector2(width, height));
        _computeShader.SetInt("g_time_ms", Time.frameCount);
        _computeShader.SetInt("g_samples_per_pixel", pathSamples);
        _computeShader.SetMatrix("g_worldToTarget", Matrix4x4.identity);
        _computeShader.SetFloat("g_TransmissibilityVariationEpsilon", transmissibilityVariationEpsilon);
        _computeShader.SetFloat("g_lightEmissionOutscatter", 0);
        _computeShader.SetInt("g_density_granularity", densityGranularity);
        _computeShader.SetShaderFlag("BILINEAR_PHOTON_DISTRIBUTION", bilinearPhotonWrites);
        
        _forwardIntegrator.ImportanceSamplingTarget = ImportanceSamplingTarget;
        _forwardIntegrator.DisableBilinearWrites = !bilinearPhotonWrites;
        _backwardIntegrator.ImportanceSamplingTarget = ImportanceSamplingTarget;

        for (int i = 0; i < _gridCellInput.Length; i++)
        {
            if (_gridCellInput[i].IsActive != 0)
            {
                _gridCellInput[i].FrameCount++;
            }
        }
        _gridCellInputBuffer.SetData(_gridCellInput);

        switch (strategy)
        {
            case Strategy.LightTransport:
                // Forward light tracing technique:
                // Photons are simulated from each light source.

                // At each collision point, a portion of the light's energy is "reflected" out to the viewer's eye (outscattered).

                _forwardIntegrator.IntegrationInterval = integrationInterval;
                _forwardIntegrator.WorldToTargetTransform = worldToTargetSpace;
                _forwardIntegrator.FramesSinceClear = framesSinceClear;
                _forwardIntegrator.ForcedBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
                _forwardIntegrator.RaysToEmit = raysPerFrame;

                _forwardIntegrator.Integrate(_allLights);
                SimulationOutputHDR = _forwardIntegrator.OutputImageHDR;
                break;
            case Strategy.Hybrid:
                // Hybrid forward/backward tracing technique

                // Photons are simulated from each light source into the simulated field for N bounces.
                //    This process is the same as it is for the LightTransport strategy except the render
                //    target is an intermediate buffer.

                // Then, paths are traced backward from view pixels through M bounces.
                //    When paths intersect a pixel that has energy deposited from a previous step,
                //    that energy is propagated to the tracing pixel.

                _forwardIntegrator.IntegrationInterval = integrationInterval;
                _forwardIntegrator.WorldToTargetTransform = worldToTargetSpace;
                _forwardIntegrator.FramesSinceClear = framesSinceClear;
                _forwardIntegrator.ForcedBounceCount = photonBounces == -1 ? null : (uint)photonBounces;
                _forwardIntegrator.RaysToEmit = raysPerFrame;

                _forwardIntegrator.Integrate(_allLights);

                _backwardIntegrator.IntegrationInterval = integrationInterval;

                _backwardIntegrator.Integrate();

                SimulationOutputHDR = _backwardIntegrator.OutputImage;
                break;
        }

        // Generate photon count and HDR mip levels
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
        OnStep?.Invoke(framesSinceClear);

        // CONVERGENCE TESTING
        bool fireConvergedEvent = false;
        if (frameLimit != -1 && framesSinceClear >= frameLimit)
        {
            if (framesSinceClear > frameLimit)
            {
                Debug.LogError("Skipped a frame somehow...");
            }

            hasConverged = true;
            fireConvergedEvent = true;
        }

        if (ConvergenceMeasurementInterval != 0 && framesSinceClear % ConvergenceMeasurementInterval == 0 ||
            framesSinceClear == 1 && _convergenceThreshold > 0)
        {
            MeasureConvergence(framesSinceClear == 1);
        }

        if (fireConvergedEvent)
        {
            OnConverged?.Invoke();
        }
    }

    async void MeasureConvergence(bool initial)
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

        _gridCellInputBuffer.SetData(_gridCellInput);

        _computeShader.SetVector("g_convergenceCells", new Vector2(gridCells.x, gridCells.y));
        _computeShader.SetFloat("g_getCellTransmissibility_lod", GBuffer.Transmissibility.mipmapCount
            - Mathf.Ceil(Mathf.Log(Mathf.Max(gridCells.x, gridCells.y), 2)) - 1);

        if (true || !hasValidGridTransmissibility)
        {
            _computeShader.RunKernel("GetCellTransmissibility", (int)gridCells.x, (int)gridCells.y,
                ("g_transmissibility", GBuffer.Transmissibility),
                ("g_convergenceCellStateOut", _gridCellOutputBuffer));
            hasValidGridTransmissibility = true;
        }

        // TODO: Measure convergence via variance map
        // _computeShader.RunKernel("MeasureConvergence", width, height,
        //     ("g_output_raw", SimulationPhotonsRaw),
        //     ("g_output_tonemapped", _renderTexture[_currentRenderTextureIndex]),
        //     ("g_previousResult", _renderTexture[1 - _currentRenderTextureIndex]),
        //     ("g_convergenceCellStateIn", _gridCellInputBuffer),
        //     ("g_convergenceCellStateOut", _gridCellOutputBuffer));
        _currentRenderTextureIndex = 1 - _currentRenderTextureIndex;

        int recentSceneId = _sceneId;
        var r = await AsyncGPUReadback.RequestAsync(_gridCellOutputBuffer);

        if (recentSceneId != _sceneId) return;
        awaitingConvergenceResult = false;
        if (!r.done || r.hasError) return;

#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying) return;
#endif

        var feedback = r.GetData<ConvergenceCellOutput>(0);

        int totalConverged = 0;
        float overallConvergence = 0;
        float cellArea = width * height / (gridCells.x * gridCells.y);
        float[] efficiencies = new float[feedback.Length];
        float minEfficiency = float.MaxValue;
        int minEfficiencyIndex = -1;
        ulong cumulativePhotons = 0;
        for (int i = 0; i < feedback.Length; i++)
        {
            var outputState = feedback[i];
            _gridCellOutput[i] = outputState;

            if (outputState.PhotonCount == 0)
                continue;

            _gridCellState[i].CumulativePhotonCount += outputState.PhotonCount;
            cumulativePhotons += _gridCellState[i].CumulativePhotonCount;

            float averagePhotonCount = _gridCellState[i].CumulativePhotonCount / (float)_gridCellInput[i].FrameCount;
            float efficiency = averagePhotonCount / ((1 - outputState.Transmissibility) * cellArea * ConvergenceMeasurementInterval);
            efficiencies[i] = efficiency;

            float flux = _gridCellState[i].CumulativePhotonCount / cellArea;

            var localConvergence = outputState.PixelChange / (ConvergenceMeasurementInterval * efficiency);
            //var localConvergence = 1000 / flux;
            bool localHasConverged = localConvergence < _convergenceThreshold;

            if (_gridCellInput[i].IsActive != 0)
            {
                //_gridCellInput[i].IsActive = localHasConverged ? 0u : 1u;
                //AccumulatePhotons(i % (int)gridCells.x, i / (int)gridCells.x);
                //_gridCellInput[i].IsActive = outputState.MaxValue > (1u << 31) ? 0u : 1u;
                overallConvergence = Math.Max(overallConvergence, localConvergence);
            }

            if (localHasConverged)
            {
                totalConverged++;
            }

            if (_gridCellInput[i].IsActive != 0 && efficiency < minEfficiency)
            {
                minEfficiency = efficiency;
                minEfficiencyIndex = i;
            }
        }

        { // Figure out photon writes per second
            float currentTime = Time.time;

            if (_previousConvergenceFeedbackTime != 0)
            {
                ulong photonDelta = cumulativePhotons - _previousCumulativePhotons;
                double dt = (double)(currentTime - _previousConvergenceFeedbackTime);
                PhotonWritesPerSecond = (uint)(photonDelta / dt);
            }

            _previousCumulativePhotons = cumulativePhotons;
            _previousConvergenceFeedbackTime = currentTime;
        }

        { // target the lowest efficiency cell
            Vector2 targetCell;
            {
                if (minEfficiencyIndex == -1)
                {
                    targetCell = new Vector2(gridCells.x / 2.0f, gridCells.y / 2.0f);
                }
                else
                {
                    int xTargetCell = minEfficiencyIndex % (int)gridCells.x;
                    int yTargetCell = minEfficiencyIndex / (int)gridCells.x;

                    targetCell = new Vector2
                    {
                        x = xTargetCell + 0.5f,
                        y = yTargetCell + 0.5f
                    };
                }

                var currentIdealTarget = new Vector2(
                    targetCell.x / gridCells.x * width,
                    targetCell.y / gridCells.y * height
                );

                ImportanceSamplingTarget = currentIdealTarget;
            }

            for (int i = 0; i < gridCells.x; i++)
            {
                for (int j = 0; j < gridCells.y; j++)
                {
                    long index = i + j * gridCells.x;

                    // Accumulate net transmissibility from cell[i,j] to targetCell
                    Vector2 originCell = new Vector2
                    {
                        x = i + 0.5f,
                        y = j + 0.5f
                    };

                    Vector2 direction = targetCell - originCell;
                    float distance = direction.magnitude;
                    direction.Normalize();

                    int xDir = direction.x < 0 ? -1 : direction.x > 0 ? 1 : 0;
                    int yDir = direction.y < 0 ? -1 : direction.y > 0 ? 1 : 0;

                    int marchX = i;
                    int marchY = j;

                    float xBoundary = i + Math.Max(xDir, 0);
                    int yBoundary = j + Math.Max(yDir, 0);

                    float uCurrent = 0;
                    float uNextX = xDir == 0 ? float.MaxValue : (xBoundary - originCell.x) / direction.x;
                    float uNextY = yDir == 0 ? float.MaxValue : (yBoundary - originCell.y) / direction.y;
                    float netTransmissibility = 1;

                    while (uCurrent < distance)
                    {
                        float cellDistance;
                        float uNext;
                        float cellTransmissibility = _gridCellOutput[marchX + gridCells.x * marchY].Transmissibility;

                        if (uNextX < uNextY)
                        {
                            cellDistance = uNextX - uCurrent;
                            uNext = uNextX;
                            marchX += xDir;
                            xBoundary += xDir;
                            uNextX = (xBoundary - originCell.x) / direction.x;
                        }
                        else
                        {
                            cellDistance = uNextY - uCurrent;
                            uNext = uNextY;
                            marchY += yDir;
                            yBoundary += yDir;
                            uNextY = (yBoundary - originCell.y) / direction.y;
                        }

                        if (uNext > distance)
                        {
                            uNext = distance;
                        }

                        netTransmissibility *= Mathf.Pow(cellTransmissibility, uNext - uCurrent);
                        uCurrent = uNext;
                    }

                    _gridCellInput[index].TargetTransmission = netTransmissibility;
                    _gridCellInput[index].TransmissionSpread = 0.9f;

                    // TargetTransmission should be an estimate of the net transmissibility
                    // from the origin cell to the target cell.
                    // TransmissionSpread should be determined by estimating the likelihood of escape and comparing that to the target.
                    // Or it can be estimated from the cell size.
                    // Or it can be constant, might work ok.
                    // 
                    //_gridCellInput[index].TargetTransmission
                }
            }
        }

        for (int i = 0; i < gridCells.x; i++)
        {
            for (int j = 0; j < gridCells.y; j++)
            {
                EfficiencyData[i, j] = efficiencies[j * gridCells.x + i];
            }
        }

        for (int i = 0; i < gridCells.x; i++)
        {
            for (int j = 0; j < gridCells.y; j++)
            {
                var g = new Vector2
                {
                    x = (EfficiencyData.ReadMirrored(i + 1, j) - EfficiencyData.ReadMirrored(i - 1, j)) / -2.0f,
                    y = (EfficiencyData.ReadMirrored(i, j + 1) - EfficiencyData.ReadMirrored(i, j - 1)) / -2.0f
                };

                g *= g.sqrMagnitude;

                EfficiencyGradient[i, j] = g;
            }
        }

        for (int i = 0; i < gridCells.x; i++)
        {
            for (int j = 0; j < gridCells.y; j++)
            {
                RelaxedEfficiencyGradient[i, j] = EfficiencyGradient[i, j];
            }
        }

        const float w = 0.4f;
        const int JacobianIterations = 1000;

        for (int n = 0; n < JacobianIterations; n++)
        {
            for (int i = 0; i < gridCells.x; i++)
            {
                for (int j = 0; j < gridCells.y; j++)
                {
                    RelaxedEfficiencyGradient[i, j] = (1 - w) * EfficiencyGradient[i, j] + w / 4.0f * (
                        RelaxedEfficiencyGradient.ReadClamped(i + 1, j) + RelaxedEfficiencyGradient.ReadClamped(i - 1, j) +
                        RelaxedEfficiencyGradient.ReadClamped(i, j + 1) + RelaxedEfficiencyGradient.ReadClamped(i, j - 1));
                }
            }
        }

        efficiencies.Visualize(EfficiencyDiagnostic);
        feedback.Select(f => (float)f.PhotonCount).Visualize(PhotonsDiagnostic);
        feedback.Select(f => (float)f.MaxValue / (float)(1u << 31)).Visualize(MaxValueDiagnostic, false);

        _gridCellOutputBuffer.SetData(_gridCellOutputInitialValue);
        _gridCellInputBuffer.SetData(_gridCellInput);
        IdentifyCellInputGroups();

        convergenceProgress = overallConvergence / cellArea;
        OnConvergenceUpdate?.Invoke(convergenceProgress);

        if (!initial && convergenceProgress < _convergenceThreshold)
        {
            hasConverged = true;
            OnConverged?.Invoke();
        }
    }

    // void AccumulatePhotons(int xCell, int yCell)
    // {
    //     int w = (int)(width / gridCells.x);
    //     int h = (int)(height / gridCells.y);

    //     _computeShader.SetVector("g_accumulate_base_index", new Vector2(xCell * w, yCell * h));
    //     _computeShader.RunKernel("AccumulatePhotons", w, h,
    //         ("g_output_accumulated", SimulationForwardAccumulated),
    //         ("g_output_raw", SimulationPhotonsRaw),
    //         ("g_convergenceCellStateIn", _gridCellInputBuffer));
    // }

    void IdentifyCellInputGroups()
    {
        _gridCellInputGroups.Clear();

        // Case 0:
        //    1 group: the whole image.
        _gridCellInputGroups.Add(new ConvergenceCellGroup
        {
            StartX = 0,
            StartY = 0,
            EndX = (int)gridCells.x,
            EndY = (int)gridCells.y
        });

        // Case 1:
        //    n*n groups, one cell per group
        // Case 2:
        //    Simple Adjacency
        // Case 3:
        //    ??

        // Find an active, ungrouped cell
        // Find the furthest extents that the group could start/end at, while being exclusive from other groups
        // while the % of active cells in the group is < 50, shrink the group size
    }
}
