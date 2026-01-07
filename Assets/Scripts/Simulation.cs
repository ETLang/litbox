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
    public int resolution;
    public int raysPerFrame;
    public int photonBounces;
    public float transmissibilityVariationEps;
    public float outscatterCoefficient;
}

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class Simulation : DisposalHelperComponent
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
        PathTracing,
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

    [SerializeField] private int raysPerFrame = 512000;
    [SerializeField] private int photonBounces = -1;
    [SerializeField] private IntegrationMethod integrationMethod = IntegrationMethod.ExplicitBoundedBounceImplicitInterval;
    [SerializeField] private float integrationInterval = 0.1f;
    [SerializeField] private int densityGranularity = 10000;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;
    [SerializeField, Range(0, 0.5f)] private float outscatterCoefficient = 0.01f;
    
    [Header("Tone Mapping")]
    [SerializeField] public bool enableToneMappping = true;
    [SerializeField] public float exposure = 0.0f;
    [SerializeField] public Vector3 whitePointLog = new Vector3(2.0f, 2.0f, 2.0f);
    [SerializeField] public Vector3 blackPointLog = new Vector3(-4.0f, -4.0f, -5.0f);

    [Header("Convergence Information")]
    [SerializeField] private float _convergenceThreshold = -1;
    [SerializeField] private int framesSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    [Header("Archaic Properties")]
    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private uint2 gridCells = new uint2(8, 8);
    [SerializeField] private int energyPrecision = 100;
    [SerializeField] private bool bilinearPhotonWrites = true;
    [SerializeField] private int pathSamples = 10;
    [SerializeField, Range(0, 1)] private float pathBalance = 0.5f;

    private static int _MainTexID = Shader.PropertyToID("_MainTex");
    private Material _compositorMat;
    private SimulationCamera _realContentCamera;
    private ComputeShader _computeShader;
    private ComputeBuffer _randomBuffer;
    private ComputeBuffer _gridCellInputBuffer;
    private ComputeBuffer _gridCellOutputBuffer;
    private List<ConvergenceCellGroup> _gridCellInputGroups = new List<ConvergenceCellGroup>();
    private ConvergenceCellInput[] _gridCellInput;
    private ConvergenceCellInput[] _gridCellInputInitialValue;
    private ConvergenceCellOutput[] _gridCellOutput;
    private ConvergenceCellOutput[] _gridCellOutputInitialValue;
    private ConvergenceCellState[] _gridCellState;

    private RenderTexture[] _renderTexture = new RenderTexture[2];
    private int _currentRenderTextureIndex = 0;
    private RenderBuffer[][] _gBuffer;
    private Texture _mieScatteringLUT;
    private Texture _teardropScatteringLUT;
    private Texture _bdrfLUT;
    private int[] _kernelsHandles;

    private bool hasValidGridTransmissibility = false;
    private bool awaitingConvergenceResult = false;
    [SerializeField, ReadOnly] public bool hasConverged = false;

    private uint4[] convergenceResultResetData = new uint4[] { new uint4(0, 0, 0, 0) };
    private SortedDictionary<float, uint> performanceCounter = new SortedDictionary<float, uint>();

    public uint TraversalsPerSecond { get; private set; }
    public uint PhotonWritesPerSecond { get; private set; }
    private ulong _previousCumulativePhotons;
    private float _previousConvergenceFeedbackTime;

    private RenderTexture[] _gBufferAlbedo = new RenderTexture[2];
    private RenderTexture[] _gBufferTransmissibility = new RenderTexture[2];
    private RenderTexture[] _gBufferNormalAlignment = new RenderTexture[2];
    private int _gBufferNextTarget = 0;
    private bool _validationFailed = false;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;

    public RenderTexture GBufferAlbedo { get; private set; }
    public RenderTexture GBufferTransmissibility { get; private set; }
    public RenderTexture GBufferNormalAlignment { get; private set; }
    public RenderTexture GBufferQuadTreeLeaves { get; private set; }

    public RenderTexture SimulationPhotonsForward { get; private set; }
    public RenderTexture SimulationOutputRaw { get; private set; }
    public RenderTexture SimulationOutputAccumulated { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
    public RenderTexture SimulationOutputToneMapped { get; private set; }
    public RenderTexture PhotonDensityBuffer { get; private set; }
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

    public void LoadProfile(SimulationProfile profile)
    {
        frameLimit = profile.frameLimit;
        width = profile.resolution;
        height = profile.resolution;
        raysPerFrame = profile.raysPerFrame;
        transmissibilityVariationEpsilon = profile.transmissibilityVariationEps;
        outscatterCoefficient = profile.outscatterCoefficient;
        photonBounces = profile.photonBounces;
        hasConverged = false;
        framesSinceClear = 0;
    }

    void ValidateRandomBuffer()
    {
        var randSeeds = Math.Max(raysPerFrame, width * height);

        if (_randomBuffer == null || _randomBuffer.count < randSeeds)
        {
            uint4[] seeds = new uint4[randSeeds];

            for (int i = 0; i < seeds.Length; i++)
            {
                seeds[i].x = (uint)(UnityEngine.Random.value * 1000000);
                seeds[i].y = (uint)(UnityEngine.Random.value * 1000000);
                seeds[i].z = (uint)(UnityEngine.Random.value * 1000000);
                seeds[i].w = (uint)(UnityEngine.Random.value * 1000000);
            }

            if (_randomBuffer != null)
            {
                _randomBuffer.Release();
            }

            _randomBuffer = this.CreateStructuredBuffer(seeds);
        }
    }

    void ValidateTargets()
    {
        if (_renderTexture[0] == null || _renderTexture[0].width != width || _renderTexture[0].height != height)
        {
            CreateTargetBuffers();
            _validationFailed = true;
        }
    }

    private void Awake()
    {
        _computeShader = (ComputeShader)Resources.Load("Simulation");

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

    void OnDestroy()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged -= EditorApplication_playModeStateChanged;
#endif
    }

#if UNITY_EDITOR
    private void EditorApplication_playModeStateChanged(UnityEditor.PlayModeStateChange state)
    {
        if(state == UnityEditor.PlayModeStateChange.EnteredPlayMode) {
            OnStartedPlaying();
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

        _mieScatteringLUT = LUT.CreateMieScatteringLUT().AsTexture();
        DisposeOnDisable(() => DestroyImmediate(_mieScatteringLUT));
        _teardropScatteringLUT = LUT.CreateTeardropScatteringLUT(10).AsTexture();
        DisposeOnDisable(() => DestroyImmediate(_teardropScatteringLUT));
        _bdrfLUT = LUT.CreateBDRFLUT().AsTexture();
        DisposeOnDisable(() => DestroyImmediate(_bdrfLUT));

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

        SwapGBuffer();
    }

    private void Start()
    {
        if(!Application.isEditor) {
            OnStartedPlaying();
        }
    }

    private void CreateTargetBuffers()
    {
        for (int i = 0; i < _renderTexture.Length; i++)
        {
            _renderTexture[i] = this.CreateRWTexture(width, height, RenderTextureFormat.DefaultHDR);
        }

        SimulationPhotonsForward = this.CreateRWTexture(width * 3, height, RenderTextureFormat.RInt);
        SimulationOutputRaw = this.CreateRWTexture(width * 3, height, RenderTextureFormat.RInt);
        SimulationOutputAccumulated = this.CreateRWTexture(width, height, RenderTextureFormat.ARGBFloat);
        SimulationOutputHDR = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat);
        PhotonDensityBuffer = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.RInt);

        for (int i = 0; i < 2; i++)
        {
            _gBufferAlbedo[i] = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat, 32);
            _gBufferTransmissibility[i] = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat);
            _gBufferNormalAlignment[i] = this.CreateRWTextureWithMips(width, height, RenderTextureFormat.ARGBFloat);
        }

        GBufferQuadTreeLeaves = this.CreateRWTexture(width, height, RenderTextureFormat.ARGBHalf);

        SwapGBuffer();
    }

    private void SwapGBuffer()
    {
        GBufferAlbedo = _gBufferAlbedo[_gBufferNextTarget];
        GBufferTransmissibility = _gBufferTransmissibility[_gBufferNextTarget];
        GBufferNormalAlignment = _gBufferNormalAlignment[_gBufferNextTarget];

        //_gBufferNextTarget = 1 - _gBufferNextTarget;

        if(_realContentCamera != null) {
            _realContentCamera.GBufferAlbedo = _gBufferAlbedo[_gBufferNextTarget];
            _realContentCamera.GBufferTransmissibility = _gBufferTransmissibility[_gBufferNextTarget];
            _realContentCamera.GBufferNormalSlope = _gBufferNormalAlignment[_gBufferNextTarget];
            _realContentCamera.GBufferQuadTreeLeaves = GBufferQuadTreeLeaves;
            _realContentCamera.VarianceEpsilon = transmissibilityVariationEpsilon;
        }
    }

    void OnEnable()
    {
        ValidateTargets();
        ValidateRandomBuffer();
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

    Matrix4x4 _previousSimulationMatrix;
    HashSet<RTLightSource> _previousLightSources = new HashSet<RTLightSource>();
    HashSet<RTObject> _previousObjects = new HashSet<RTObject>();
    float _previousOutscatterCoefficient;
    float _previousPathBalance;
    int _sceneId;
    IntegrationMethod _previousIntegrationMethod;

    bool CheckChanged(RTLightSource[] allLights, RTObject[] allObjects, Matrix4x4 worldToPresentation)
    {
        var changed =
            allLights.Length != _previousLightSources.Count ||
            !allLights.All(l => _previousLightSources.Contains(l)) ||
            allLights.Any(l => l.Changed) ||
            allObjects.Length != _previousObjects.Count ||
            !allObjects.All(o => _previousObjects.Contains(o)) ||
            allObjects.Any(o => o.Changed) ||
            _previousSimulationMatrix != worldToPresentation ||
            _previousOutscatterCoefficient != outscatterCoefficient ||
            _previousPathBalance != pathBalance ||
            _previousIntegrationMethod != integrationMethod;

        _previousIntegrationMethod = integrationMethod;
        _previousPathBalance = pathBalance;
        _previousOutscatterCoefficient = outscatterCoefficient;
        _previousSimulationMatrix = worldToPresentation;
        _previousLightSources.Clear();
        foreach (var light in allLights)
            _previousLightSources.Add(light);
        _previousObjects.Clear();
        foreach (var o in allObjects)
            _previousObjects.Add(o);

        return changed;
    }

    void Update()
    {
        ValidateTargets();
        ValidateRandomBuffer();
    }

    const int ConvergenceMeasurementInterval = 100;
    void LateUpdate()
    {
        if (_realContentCamera == null)
        {
            return;
        }

        _realContentCamera.Render();

        var worldToPresentation = _realContentCamera.transform.worldToLocalMatrix;
        var presentationToTargetSpace = Matrix4x4.Scale(new Vector3(width, height, 1)) * Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        //var presentationToTargetSpace = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        var worldToTargetSpace = presentationToTargetSpace * worldToPresentation;
        var allLights = FindObjectsByType<RTLightSource>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        var allObjects = FindObjectsByType<RTObject>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);

        // PERFORMANCE MEASUREMENT
        var now = Time.time;
        while (performanceCounter.Keys.Count != 0 && performanceCounter.Keys.First() < now - 1)
            performanceCounter.Remove(performanceCounter.Keys.First());

        uint total = 0;
        foreach (var value in performanceCounter.Values)
            total += value;
        uint bouncesThisFrame = 0;

        foreach (var bounces in allLights.Select(light => light.bounces))
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
        if (CheckChanged(allLights, allObjects, worldToPresentation) || _validationFailed)
        {
            hasConverged = false;
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

            SimulationOutputRaw.Clear(Color.clear);
            SimulationOutputAccumulated.Clear(Color.clear);
            PhotonDensityBuffer.Clear(Color.clear);

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
        _computeShader.SetInt("g_lowest_lod", (int)(GBufferTransmissibility.mipmapCount - 4));
        _computeShader.SetInt("g_4x4_lod", (int)(GBufferTransmissibility.mipmapCount - 3));
        _computeShader.SetFloat("g_lightEmissionOutscatter", 0);
        _computeShader.SetInt("g_density_granularity", densityGranularity);
        _computeShader.SetFloat("g_outscatterCoefficient", outscatterCoefficient);
        _computeShader.SetShaderFlag("BILINEAR_PHOTON_DISTRIBUTION", bilinearPhotonWrites);

        _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT", false);
        _computeShader.SetShaderFlag("INTEGRATE_IMPLICIT", false);
        _computeShader.SetShaderFlag("INTEGRATE_IMPLICIT_INTERVAL", false);
        _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNDED", false);
        _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNCE_IMPLICIT_INTERVAL", false);
        _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNDED_BOUNCE_IMPLICIT_INTERVAL", false);

        switch (integrationMethod)
        {
            case IntegrationMethod.Explicit:
                _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT", true);
                break;
            case IntegrationMethod.Implicit:
                _computeShader.SetShaderFlag("INTEGRATE_IMPLICIT", true);
                break;
            case IntegrationMethod.ImplicitInterval:
                _computeShader.SetShaderFlag("INTEGRATE_IMPLICIT_INTERVAL", true);
                break;
            case IntegrationMethod.ExplicitBounded:
                _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNDED", true);
                break;
            case IntegrationMethod.ExplicitBounceImplicitInterval:
                _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNCE_IMPLICIT_INTERVAL", true);
                break;
            case IntegrationMethod.ExplicitBoundedBounceImplicitInterval:
                _computeShader.SetShaderFlag("INTEGRATE_EXPLICIT_BOUNDED_BOUNCE_IMPLICIT_INTERVAL", true);
                break;
        }
        

        float energyNormPerFrame = 1;
        float pixelCount = width * height;

        switch (strategy)
        {
            case Strategy.LightTransport:
                // Forward light tracing technique:
                // Photons are simulated from each light source.

                // At each collision point, a portion of the light's energy is "reflected" out to the viewer's eye (outscattered).

                /*
                Gridcell management:
                For each cell group, compute center and ideal lobe size.
                Simulate lights
                Filter cells that have already converged.

                */
                energyNormPerFrame = raysPerFrame / pixelCount;
                _computeShader.SetFloat("g_energy_norm", (float)((double)uint.MaxValue / pixelCount * energyPrecision));

                _computeShader.SetShaderFlag("FILTER_INACTIVE_CELLS", true);
                foreach (var light in allLights)
                {
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
                SimulationPhotonsForward.Clear(Color.clear);
                break;
            case Strategy.Hybrid:
                // Hybrid forward/backward tracing technique

                // Photons are simulated from each light source into the simulated field for N bounces.
                //    This process is the same as it is for the LightTransport strategy except the render
                //    target is an intermediate buffer.

                // Then, paths are traced backward from view pixels through M bounces.
                //    When paths intersect a pixel that has energy deposited from a previous step,
                //    that energy is propagated to the tracing pixel.

                energyNormPerFrame = raysPerFrame / pixelCount;
                _computeShader.SetFloat("g_energy_norm", (float)((double)uint.MaxValue / pixelCount * energyPrecision));
                _computeShader.SetFloat("g_path_balance", pathBalance);

                // Clear intermediate target
                SimulationPhotonsForward.Clear(Color.clear);
                _computeShader.SetShaderFlag("FILTER_INACTIVE_CELLS", false);
                foreach (var light in allLights)
                {
                    SimulateLight(light, Strategy.LightTransport, photonBounces != -1 ? photonBounces / 2 : (int)light.bounces, worldToTargetSpace, SimulationPhotonsForward);
                }

                _computeShader.SetShaderFlag("FILTER_INACTIVE_CELLS", true);
                _computeShader.RunKernel("Simulate_View_Backward", width, height,
                    ("g_rand", _randomBuffer),
                    ("g_photons_forward", SimulationPhotonsForward),
                    ("g_output_raw", SimulationOutputRaw),
                    ("g_albedo", GBufferAlbedo),
                    ("g_transmissibility", GBufferTransmissibility),
                    ("g_normalAlignment", GBufferNormalAlignment),
                    ("g_quadTreeLeaves", GBufferQuadTreeLeaves),
                    ("g_mieScatteringLUT", _mieScatteringLUT),
                    ("g_teardropScatteringLUT", _teardropScatteringLUT),
                    ("g_bdrfLUT", _bdrfLUT),
                    ("g_convergenceCellStateIn", _gridCellInputBuffer),
                    ("g_convergenceCellStateOut", _gridCellOutputBuffer));
                break;
        }

        for (int i = 0; i < _gridCellInput.Length; i++)
        {
            if (_gridCellInput[i].IsActive != 0)
            {
                _gridCellInput[i].FrameCount++;
            }
        }
        _gridCellInputBuffer.SetData(_gridCellInput);

        // HDR MAPPING
        _computeShader.RunKernel("ConvertToHDR", width, height,
            ("g_output_raw", SimulationOutputRaw),
            ("g_output_accumulated", SimulationOutputAccumulated),
            ("g_output_hdr", SimulationOutputHDR),
            ("g_convergenceCellStateIn", _gridCellInputBuffer));

        // Generate photon count and HDR mip levels
        int mipW = PhotonDensityBuffer.width;
        int mipH = PhotonDensityBuffer.height;
        for(int i = 1;i < PhotonDensityBuffer.mipmapCount;i++) {
            mipW /= 2;
            mipH /= 2;
            _computeShader.RunKernel("GenerateOutputMips", mipW, mipH,
                ("g_sourceMipLevelPhotonCount", PhotonDensityBuffer.SelectMip(i - 1)),
                ("g_sourceMipLevelHDR", SimulationOutputHDR.SelectMip(i - 1)),
                ("g_destMipLevelPhotonCount", PhotonDensityBuffer.SelectMip(i)),
                ("g_destMipLevelHDR", SimulationOutputHDR.SelectMip(i)));
        }

        // TONE MAPPING
        // TODO: Leverage adaptive sampling from photon count buffer
        if (enableToneMappping)
        {
            _computeShader.RunKernel("ToneMap", width, height,
                ("g_hdr", SimulationOutputHDR),
                ("g_photon_density", PhotonDensityBuffer),
                ("g_output_tonemapped", _renderTexture[_currentRenderTextureIndex]),
                ("g_convergenceCellStateIn", _gridCellInputBuffer),
                ("g_blackPointLog", blackPointLog),
                ("g_whitePointLog", whitePointLog),
                ("g_exposure", exposure));
        } else {
            SimulationOutputToneMapped = SimulationOutputHDR;
        }

        SimulationOutputToneMapped = _renderTexture[_currentRenderTextureIndex];
        _compositorMat.SetTexture(_MainTexID, SimulationOutputToneMapped);

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

        SwapGBuffer();

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
        _computeShader.SetFloat("g_getCellTransmissibility_lod", GBufferTransmissibility.mipmapCount
            - Mathf.Ceil(Mathf.Log(Mathf.Max(gridCells.x, gridCells.y), 2)) - 1);

        if (true || !hasValidGridTransmissibility)
        {
            _computeShader.RunKernel("GetCellTransmissibility", (int)gridCells.x, (int)gridCells.y,
                ("g_transmissibility", GBufferTransmissibility),
                ("g_convergenceCellStateOut", _gridCellOutputBuffer));
            hasValidGridTransmissibility = true;
        }

        _computeShader.RunKernel("MeasureConvergence", width, height,
            ("g_output_raw", SimulationOutputRaw),
            ("g_output_tonemapped", _renderTexture[_currentRenderTextureIndex]),
            ("g_previousResult", _renderTexture[1 - _currentRenderTextureIndex]),
            ("g_convergenceCellStateIn", _gridCellInputBuffer),
            ("g_convergenceCellStateOut", _gridCellOutputBuffer));
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
                AccumulatePhotons(i % (int)gridCells.x, i / (int)gridCells.x);
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

        //float BoundaryEfficiency = 3;

        for (int i = 0; i < gridCells.x; i++)
        {
            for (int j = 0; j < gridCells.y; j++)
            {
                var g = new Vector2
                {
                    // x = (EfficiencyData.ReadClampScaled(i+1,j,BoundaryEfficiency) - EfficiencyData.ReadClampScaled(i-1,j,BoundaryEfficiency)) / -2.0f,
                    // y = (EfficiencyData.ReadClampScaled(i,j+1,BoundaryEfficiency) - EfficiencyData.ReadClampScaled(i,j-1,BoundaryEfficiency)) / -2.0f
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

    void AccumulatePhotons(int xCell, int yCell)
    {
        int w = (int)(width / gridCells.x);
        int h = (int)(height / gridCells.y);

        _computeShader.SetVector("g_accumulate_base_index", new Vector2(xCell * w, yCell * h));
        _computeShader.RunKernel("AccumulatePhotons", w, h,
            ("g_output_accumulated", SimulationOutputAccumulated),
            ("g_output_raw", SimulationOutputRaw),
            ("g_convergenceCellStateIn", _gridCellInputBuffer));
    }

    void SimulateLight(RTLightSource light, Strategy strategy, int bounces, Matrix4x4 worldToTargetSpace, RenderTexture outputTexture)
    {
        string simulateKernel = null;
        int simulateKernelId = -1;
        var lightToTargetSpace = worldToTargetSpace * light.WorldTransform;
        double photonEnergy = (double)uint.MaxValue / raysPerFrame * energyPrecision;

        string kernelFormat;

        switch (strategy)
        {
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

        switch (light)
        {
            case RTPointLight pt:
                simulateKernel = string.Format(kernelFormat, "PointLight");
                _computeShader.SetFloat("g_lightEmissionOutscatter", pt.emissionOutscatter);
                break;
            case RTSpotLight _:
                simulateKernel = string.Format(kernelFormat, "SpotLight");
                break;
            case RTLaserLight _:
                simulateKernel = string.Format(kernelFormat, "LaserLight");
                break;
            case RTAmbientLight _:
                simulateKernel = string.Format(kernelFormat, "AmbientLight");
                break;
            case RTFieldLight field:
                simulateKernel = string.Format(kernelFormat, "FieldLight");
                simulateKernelId = _computeShader.FindKernel(simulateKernel);
                _computeShader.SetTexture(simulateKernelId, "g_lightFieldTexture", field.lightTexture ? field.lightTexture : Texture2D.whiteTexture);
                _computeShader.SetFloat("g_lightEmissionOutscatter", field.emissionOutscatter);
                break;
            case RTDirectionalLight dir:
                simulateKernel = string.Format(kernelFormat, "DirectionalLight");
                _computeShader.SetVector("g_directionalLightDirection", lightToTargetSpace.MultiplyVector(new Vector3(0, -1, 0)));
                break;
        }

        simulateKernelId = _computeShader.FindKernel(simulateKernel);

        _computeShader.SetVector("g_lightEnergy", light.Energy * (float)photonEnergy);
        _computeShader.SetInt("g_bounces", bounces);
        _computeShader.SetMatrix("g_lightToTarget", lightToTargetSpace.transpose);
        _computeShader.SetFloat("g_integration_interval", Mathf.Max(0.01f, integrationInterval * height));

        _computeShader.RunKernel(simulateKernel, raysPerFrame,
            ("g_rand", _randomBuffer),
            ("g_output_raw", outputTexture),
            ("g_photon_density_raw", PhotonDensityBuffer),
            ("g_albedo", GBufferAlbedo),
            ("g_transmissibility", GBufferTransmissibility),
            ("g_normalAlignment", GBufferNormalAlignment),
            ("g_quadTreeLeaves", GBufferQuadTreeLeaves),
            ("g_mieScatteringLUT", _mieScatteringLUT),
            ("g_teardropScatteringLUT", _teardropScatteringLUT),
            ("g_bdrfLUT", _bdrfLUT),
            ("g_convergenceCellStateIn", _gridCellInputBuffer),
            ("g_convergenceCellStateOut", _gridCellOutputBuffer));
    }

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
