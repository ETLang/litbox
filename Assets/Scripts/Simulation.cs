using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Linq;
using System.Threading;
using Mono.Cecil.Cil;


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

    struct ConvergenceCellInput {
        public uint IsActive;
        public uint HasConverged;
        public uint FrameCount;
        public float Reserved0;
    }

    struct ConvergenceCellOutput {
        public uint MaxValue;
        public uint PixelChange;
        public uint PhotonCount;
        public float Transmissibility;
    }

    struct ConvergenceCellGroup {
        public int StartX;
        public int StartY;
        public int EndX;
        public int EndY;
    }

    struct ConvergenceCellState {
        public uint CumulativePhotonCount;
    }

    [SerializeField] private LayerMask rayTracedLayers;

    [SerializeField] private int frameLimit = -1;
    [SerializeField] private int textureResolution = 256;

    [SerializeField] private Strategy strategy = Strategy.LightTransport;
    [SerializeField] private uint2 gridCells = new uint2(8,8);
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
    private int[] _kernelsHandles;

    private Renderer _renderer;

    [Header("Convergence Information")]
    [SerializeField] private float _convergenceThreshold = 10;
    [SerializeField] private int framesSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    private bool hasValidGridTransmissibility = false;
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

    public RenderTexture SimulationForwardPhase { get; private set; }
    public RenderTexture SimulationOutputRaw { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
    public RenderTexture SimulationOutputToneMapped { get; private set; }
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

        _gridCellState = new ConvergenceCellState[gridCells.x * gridCells.y];
        for(int i = 0;i < _gridCellState.Length;i++) {
            _gridCellState[i] = new ConvergenceCellState
            {
                CumulativePhotonCount = 0
            };
        }

        _gridCellInput = new ConvergenceCellInput[gridCells.x * gridCells.y];
        for(int i = 0;i < _gridCellInput.Length;i++) {
            _gridCellInput[i] = new ConvergenceCellInput
            {
                IsActive = 1,
                HasConverged = 0,
                FrameCount = 0
            };
        }
        _gridCellInputInitialValue = (ConvergenceCellInput[])_gridCellInput.Clone();
        _gridCellInputBuffer = CreateStructuredBuffer(_gridCellInput);

        _gridCellOutput = new ConvergenceCellOutput[gridCells.x * gridCells.y];
        for(int i = 0;i < _gridCellOutput.Length;i++) {
            _gridCellOutput[i] = new ConvergenceCellOutput
            {
                PixelChange = 0,
                PhotonCount = 0,
                MaxValue = 0
            };
        }
        _gridCellOutputInitialValue = (ConvergenceCellOutput[])_gridCellOutput.Clone();
        _gridCellOutputBuffer = CreateStructuredBuffer(_gridCellOutput);

        SimulationForwardPhase = CreateRWTexture(textureResolution * 3, textureResolution, RenderTextureFormat.RInt);
        SimulationOutputRaw = CreateRWTexture(textureResolution * 3, textureResolution, RenderTextureFormat.RInt);
        SimulationOutputHDR = CreateRWTexture(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);

        for(int i = 0;i < 2;i++) {
            _gBufferAlbedo[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
            _gBufferTransmissibility[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
            _gBufferNormalSlope[i] = CreateRWTextureWithMips(textureResolution, textureResolution, RenderTextureFormat.ARGBFloat);
        }

        GBufferQuadTreeLeaves = CreateRWTexture(textureResolution, textureResolution, RenderTextureFormat.ARGBHalf);

        _mieScatteringLUT = LUT.CreateMieScatteringLUT().AsTexture();
        DisposeOnDisable(() => DestroyImmediate(_mieScatteringLUT));
        _teardropScatteringLUT = LUT.CreateTeardropScatteringLUT(4).AsTexture();
        DisposeOnDisable(() => DestroyImmediate(_teardropScatteringLUT));

        EfficiencyDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false)
        {
            filterMode = FilterMode.Point,
        };
        PhotonsDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false)
        {
            filterMode = FilterMode.Point
        };
        MaxValueDiagnostic = new Texture2D((int)gridCells.x, (int)gridCells.y, TextureFormat.RGBAFloat, false)
        {
            filterMode = FilterMode.Point
        };

        EfficiencyData = new float[gridCells.x, gridCells.y];
        EfficiencyGradient = new Vector2[gridCells.x, gridCells.y];
        for(int i = 0;i < gridCells.x;i++) {
            for(int j = 0;j < gridCells.y;j++) {
                EfficiencyGradient[i,j] = new Vector2(0,0);
            }
        }
        RelaxedEfficiencyGradient = new Vector2[gridCells.x, gridCells.y];
        for(int i = 0;i < gridCells.x;i++) {
            for(int j = 0;j < gridCells.y;j++) {
                RelaxedEfficiencyGradient[i,j] = new Vector2(0,0);
            }
        }
        //EfficiencyDiagnostic.SetPixels()

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

    const int ConvergenceMeasurementInterval = 100;
    void Update() {
        _realContentCamera?.Render();

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
            _gridCellInput = (ConvergenceCellInput[])_gridCellInputInitialValue.Clone();
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
            hasValidGridTransmissibility = false;
            awaitingConvergenceResult = false;
            convergenceProgress = -1;
            ConvergenceStartTime = now;

            SimulationOutputRaw.Clear(Color.clear);
        }

        framesSinceClear++;

        // RAY TRACING SIMULATION
        _computeShader.SetVector("g_importance_sampling_target", ImportanceSamplingTarget);
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

        _computeShader.SetVector("g_teardropScatteringLUTWindow",
            new Vector2(0.5f / _teardropScatteringLUT.width, 1 - 1.0f / _teardropScatteringLUT.width));

        float energyNormPerFrame = 1;
        float pixelCount = textureResolution * textureResolution;
        
        switch(strategy) {
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
                energyNormPerFrame = photonsPerThread * threadCount / pixelCount;
                _computeShader.SetFloat("g_energy_norm", energyNormPerFrame * energyUnit);

                SetShaderFlag(_computeShader, "FILTER_INACTIVE_CELLS", true);
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
                SimulationForwardPhase.Clear(Color.clear);
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
                _computeShader.SetFloat("g_energy_norm", energyNormPerFrame * energyUnit);
                _computeShader.SetFloat("g_path_balance", pathBalance);

                // Clear intermediate target
                SimulationForwardPhase.Clear(Color.clear);
                SetShaderFlag(_computeShader, "FILTER_INACTIVE_CELLS", false);
                foreach(var light in allLights) {
                    SimulateLight(light, Strategy.LightTransport, photonBounces != -1 ? photonBounces / 2 : (int)light.bounces, worldToTargetSpace, SimulationForwardPhase);
                }

                SetShaderFlag(_computeShader, "FILTER_INACTIVE_CELLS", true);
                RunKernel(_computeShader, "Simulate_View_Backward", textureResolution, textureResolution,
                    ("g_rand", _randomBuffer),
                    ("g_photons_forward", SimulationForwardPhase),
                    ("g_photons_final", SimulationOutputRaw),
                    ("g_albedo", GBufferAlbedo),
                    ("g_transmissibility", GBufferTransmissibility),
                    ("g_normalSlope", GBufferNormalSlope),
                    ("g_quadTreeLeaves", GBufferQuadTreeLeaves),
                    ("g_mieScatteringLUT", _mieScatteringLUT),
                    ("g_teardropScatteringLUT", _teardropScatteringLUT),
                    ("g_convergenceCellStateIn", _gridCellInputBuffer),
                    ("g_convergenceCellStateOut", _gridCellOutputBuffer));
                break;
        }

        // HDR MAPPING
        RunKernel(_computeShader, "ConvertToHDR", textureResolution, textureResolution,
            ("g_photons_final", SimulationOutputRaw),
            ("g_hdrResult", SimulationOutputHDR));

        // TONE MAPPING
        for(int i = 0;i < _gridCellInput.Length;i++) {
            if(_gridCellInput[i].IsActive != 0) {
                _gridCellInput[i].FrameCount++;
            }
        }
        _gridCellInputBuffer.SetData(_gridCellInput);

        RunKernel(_computeShader, "ToneMap", textureResolution, textureResolution,
            ("g_photons_final", SimulationOutputRaw),
            ("g_result", _renderTexture[_currentRenderTextureIndex]),
            ("g_convergenceCellStateIn", _gridCellInputBuffer));

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

        if(ConvergenceMeasurementInterval != 0 && framesSinceClear % ConvergenceMeasurementInterval == 0) {
            MeasureConvergence();
        }

        SwapGBuffer();
    }

    void MeasureConvergence() {
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

        if(awaitingConvergenceResult) return;
        awaitingConvergenceResult = true;
        if(hasConverged) return;

        _gridCellInputBuffer.SetData(_gridCellInput);

        _computeShader.SetVector("g_convergenceCells", new Vector2(gridCells.x, gridCells.y));
        _computeShader.SetFloat("g_getCellTransmissibility_lod", GBufferTransmissibility.mipmapCount -5);

        if(true || !hasValidGridTransmissibility) {
            RunKernel(_computeShader, "GetCellTransmissibility", (int)gridCells.x, (int)gridCells.y,
                ("g_transmissibility", GBufferTransmissibility),
                ("g_convergenceCellStateOut", _gridCellOutputBuffer));
            hasValidGridTransmissibility = true;
        }

        RunKernel(_computeShader, "MeasureConvergence", textureResolution, textureResolution,
            ("g_photons_final", SimulationOutputRaw),
            ("g_result", _renderTexture[_currentRenderTextureIndex]),
            ("g_previousResult", _renderTexture[1-_currentRenderTextureIndex]),
            ("g_convergenceCellStateIn", _gridCellInputBuffer),
            ("g_convergenceCellStateOut", _gridCellOutputBuffer));
        _currentRenderTextureIndex = 1 - _currentRenderTextureIndex;

        int recentSceneId = _sceneId;
        AsyncGPUReadback.Request(_gridCellOutputBuffer, (r) =>
        {
            if(recentSceneId != _sceneId) return;
            awaitingConvergenceResult = false;
            if(!r.done || r.hasError) return;

            var feedback = r.GetData<ConvergenceCellOutput>(0);

            int totalConverged = 0;
            float overallConvergence = 0;
            float cellArea = textureResolution * textureResolution / (gridCells.x * gridCells.y);
            float[] efficiencies = new float[feedback.Length];
            float minEfficiency = float.MaxValue;
            int minEfficiencyIndex = -1;
            for(int i = 0;i < feedback.Length;i++) {
                var outputState = feedback[i];
                _gridCellOutput[i] = outputState;

                if(outputState.PhotonCount == 0)
                    continue;

                _gridCellState[i].CumulativePhotonCount += outputState.PhotonCount;
                float efficiency = outputState.PhotonCount / ((1 - outputState.Transmissibility) * cellArea * ConvergenceMeasurementInterval);
                efficiencies[i] = efficiency;

                if(efficiency < minEfficiency) {
                    minEfficiency = efficiency;
                    minEfficiencyIndex = i;
                }

                float flux = _gridCellState[i].CumulativePhotonCount / cellArea;

                var localConvergence = outputState.PixelChange / (ConvergenceMeasurementInterval * efficiency);
                //var localConvergence = 1000 / flux;
                bool localHasConverged = localConvergence < _convergenceThreshold;

                if(_gridCellInput[i].IsActive != 0) {
                    //_gridCellInput[i].IsActive = localHasConverged ? 0u : 1u;
                    _gridCellInput[i].IsActive = outputState.MaxValue > (1u << 31) ? 0u : 1u;
                    overallConvergence += localConvergence;
                }

                if(localHasConverged) {
                    totalConverged++;
                }
            }

            { // target the lowest efficiency cell
                int xCell = minEfficiencyIndex % (int)gridCells.x;
                int yCell = minEfficiencyIndex / (int)gridCells.y;

                var currentIdealTarget = new Vector2(
                    (xCell + 0.5f) / gridCells.x,
                    (yCell + 0.5f) / gridCells.y
                );

                ImportanceSamplingTarget = 0.9f * ImportanceSamplingTarget + 0.1f * currentIdealTarget;
            }

            for(int i = 0;i < gridCells.x;i++) {
                for(int j = 0;j < gridCells.y;j++) {
                    EfficiencyData[i,j] = efficiencies[j * gridCells.x + i];
                }
            }

            float BoundaryEfficiency = 3;

            for(int i = 0;i < gridCells.x;i++) {
                for(int j = 0;j < gridCells.y;j++) {
                    var g = new Vector2{
                        // x = (EfficiencyData.ReadClampScaled(i+1,j,BoundaryEfficiency) - EfficiencyData.ReadClampScaled(i-1,j,BoundaryEfficiency)) / -2.0f,
                        // y = (EfficiencyData.ReadClampScaled(i,j+1,BoundaryEfficiency) - EfficiencyData.ReadClampScaled(i,j-1,BoundaryEfficiency)) / -2.0f
                        x = (EfficiencyData.ReadMirrored(i+1,j) - EfficiencyData.ReadMirrored(i-1,j)) / -2.0f,
                        y = (EfficiencyData.ReadMirrored(i,j+1) - EfficiencyData.ReadMirrored(i,j-1)) / -2.0f
                    };

                    g *= g.sqrMagnitude;

                    EfficiencyGradient[i,j] = g;
                }
            }

            for(int i = 0;i < gridCells.x;i++) {
                for(int j = 0;j < gridCells.y;j++) {
                    RelaxedEfficiencyGradient[i,j] = EfficiencyGradient[i,j];
                }
            }

            const float w = 0.4f;
            const int JacobianIterations = 1000;

            for(int n = 0;n < JacobianIterations;n++) {
                for(int i = 0;i < gridCells.x;i++) {
                    for(int j = 0;j < gridCells.y;j++) {
                        RelaxedEfficiencyGradient[i,j] = (1-w) * EfficiencyGradient[i,j] + w/4.0f * (
                            RelaxedEfficiencyGradient.ReadClamped(i+1,j) + RelaxedEfficiencyGradient.ReadClamped(i-1,j) +
                            RelaxedEfficiencyGradient.ReadClamped(i,j+1) + RelaxedEfficiencyGradient.ReadClamped(i,j-1));
                    }
                }
            }

            Visualize(EfficiencyDiagnostic, efficiencies);
            Visualize(PhotonsDiagnostic, feedback.Select(f => (float)f.PhotonCount));
            Visualize(MaxValueDiagnostic, feedback.Select(f => (float)f.MaxValue / (float)(1u << 31)), false);

            _gridCellOutputBuffer.SetData(_gridCellOutputInitialValue);
            _gridCellInputBuffer.SetData(_gridCellInput);
            IdentifyCellInputGroups();

            convergenceProgress = overallConvergence;
            if(totalConverged == feedback.Length) {
                hasConverged = true;
                OnConverged?.Invoke();
            }
        });
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
        _computeShader.SetBuffer(simulateKernel, "g_convergenceCellStateIn", _gridCellInputBuffer);
        _computeShader.SetBuffer(simulateKernel, "g_convergenceCellStateOut", _gridCellOutputBuffer);

        _computeShader.Dispatch(simulateKernel, threadCount / 64, 1, 1);
    }

    void IdentifyCellInputGroups() {
        _gridCellInputGroups.Clear();

        // Case 0:
        //    1 group: the whole image.
        _gridCellInputGroups.Add(new ConvergenceCellGroup {
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
