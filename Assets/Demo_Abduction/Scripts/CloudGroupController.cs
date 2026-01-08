using System;
using System.Linq;
using GLTFast.Materials;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;

public class CloudGroupController : PhotonerDemoComponent
{
    [SerializeField] Material foregroundCloudMat;
    [SerializeField] int foregroundSimulationLOD = 5;
    [SerializeField, Range(0, 120)] int blurSize = 1; // kernelsize = 2 * blurSize + 1
    
    Simulation _simulation;
    BindSimulationToCamera _binder;
    RenderTexture _intermediateSimulationTex;
    RenderTexture _foregroundSimulationTex;
    Matrix4x4 _simulationUVTransform;
    ComputeShader _gaussianBlurShader;

    Vector4[] _kernelSamples1 = new Vector4[256];
    Vector4[] _kernelSamples2 = new Vector4[256];
    float[] _kernelWeights;
    Texture _weightsLUT;
    int _kernelSampleCount;

    ProceduralCloud[] _clouds;

    private static int _foregroundSimulationTexId = Shader.PropertyToID("_ForegroundSimulationTex");
    private static int _foregroundSimuilationLodId = Shader.PropertyToID("_ForegroundSimulationLOD");
    private static int _foregroundSimulationUVTransformId = Shader.PropertyToID("_ForegroundSimulationUVTransform");

    public CloudGroupController()
    {
        DetectChanges(() => foregroundCloudMat);
        DetectChanges(() => foregroundSimulationLOD, "foregroundSimulation");
        DetectChanges(() => blurSize);
        DetectChanges(() => _simulationUVTransform);
        DetectChanges(() => _simulation?.SimulationOutputHDR.width, "foregroundSimulation");
        DetectChanges(() => _simulation?.SimulationOutputHDR.height, "foregroundSimulation");
    }

    void Awake()
    {
        _gaussianBlurShader = (ComputeShader)Resources.Load("GaussianBlur");
        _binder = Camera.main.GetComponentInChildren<BindSimulationToCamera>();
        _simulation = _binder.GetComponent<Simulation>();
    }

    private void OnEnable()
    {
        if(_simulation != null) {
            _simulation.OnStep += OnSimulationUpdated;
        }

        OnInvalidated("foregroundSimulation");
    }

    protected override void OnDisable()
    {
        if(_simulation != null) {
            _simulation.OnStep -= OnSimulationUpdated;
        }

        _clouds = null;
    }

    private GraphicsFence _fence;
    private void OnSimulationUpdated(int frameCount)
    {
        Vector2 pixelSize = new Vector2(1.0f / _foregroundSimulationTex.width, 1.0f / _foregroundSimulationTex.height);
        Vector2 sampleOffset = -_kernelSampleCount * pixelSize / 2.0f;

        if(blurSize != 0) {
            _gaussianBlurShader.RunKernel("GaussianBlur1D", _foregroundSimulationTex.width, _foregroundSimulationTex.height,
                ("blur_input", _simulation.SimulationOutputHDR),
                ("blur_output", _intermediateSimulationTex),
                ("kernel_lut", _weightsLUT),
                ("sample_offset", new Vector2(sampleOffset.x, 0)),
                ("sample_increment", new Vector2(pixelSize.x, 0)),
                ("lod", foregroundSimulationLOD));
            _fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

            Graphics.WaitOnAsyncGraphicsFence(_fence);
            _gaussianBlurShader.RunKernel("GaussianBlur1D", _foregroundSimulationTex.width, _foregroundSimulationTex.height,
                ("blur_input", _intermediateSimulationTex),
                ("blur_output", _foregroundSimulationTex),
                ("kernel_lut", _weightsLUT),
                ("sample_offset", new Vector2(0, sampleOffset.y)),
                ("sample_increment", new Vector2(0, pixelSize.y)),
                ("lod", 0));
            _fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

            Graphics.WaitOnAsyncGraphicsFence(_fence);
        }
    }

    protected override void Update()
    {
        base.Update();

        if(_binder != null) {
            _simulationUVTransform = _binder.ScreenToSimulationUVTransform;
        }

        if(foregroundCloudMat != null) {
            foregroundCloudMat.SetFloat("_exposure", _simulation.exposure);
            foregroundCloudMat.SetVector("_whitePointLog", _simulation.whitePointLog);
            foregroundCloudMat.SetVector("_blackPointLog", _simulation.blackPointLog);
        }
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        if(group == "foregroundSimulation") {
            if(_intermediateSimulationTex != null) {
                _intermediateSimulationTex.Release();
                DestroyImmediate(_intermediateSimulationTex);
            }

            if(_foregroundSimulationTex != null) {
                _foregroundSimulationTex.Release();
                DestroyImmediate(_foregroundSimulationTex);
            }

            _intermediateSimulationTex = this.CreateRWTexture(
                _simulation.SimulationOutputHDR.MipWidth(foregroundSimulationLOD),
                _simulation.SimulationOutputHDR.MipHeight(foregroundSimulationLOD),
                _simulation.SimulationOutputHDR.format);

            _foregroundSimulationTex = this.CreateRWTexture(
                _simulation.SimulationOutputHDR.MipWidth(foregroundSimulationLOD),
                _simulation.SimulationOutputHDR.MipHeight(foregroundSimulationLOD),
                _simulation.SimulationOutputHDR.format);
            
            _kernelSampleCount = 0; // Force kernel recalculation
        }

        if(_clouds == null && enabled) {
            _clouds = this.GetComponentsInDescendants<ProceduralCloud>().ToArray();
        }

        //var expectedKernelSize = (blurSize + 1) * (blurSize + 1);
        var expectedKernelSize = (2 * blurSize + 1);// * (2 * blurSize + 1);

        if(expectedKernelSize != _kernelSampleCount && expectedKernelSize <= 256)
        {
            _kernelSampleCount = expectedKernelSize;
            _kernelWeights = new float[_kernelSampleCount];

            int kernelSize = blurSize * 2 + 1;
            float[,] singleWeights = new float[kernelSize, kernelSize];
            float sigma = blurSize / 6.0f;
            float twoSigmaSq = 2 * sigma * sigma;
            float sqrtTwoPiSigma = Mathf.Sqrt(Mathf.PI * twoSigmaSq);
            float total = 0;
            for(int i = -blurSize;i <= blurSize;i++) {
               // for(int j = -blurSize;j <= blurSize;j++) {
                    float weight = Mathf.Exp(-((i * i) / (float)blurSize * 1.5f));
                    _kernelWeights[i + blurSize] = weight;
                    total += weight;
               // }
            }

            // normalize
            for(int i = 0;i < kernelSize;i++) {
                _kernelWeights[i] /= total;
            }

            Vector2 pixelSize = new Vector2(1.0f / _foregroundSimulationTex.width, 1.0f / _foregroundSimulationTex.height);

            int sampleIndex = 0;
            for(int i = 0;i < kernelSize;i++/*i+=2*/) {
               // for(int j = 0;j < kernelSize;j++/*j+=2*/) {

                    // float a = singleWeights[i, j];
                    // float b = (i + 1 < kernelSize) ? singleWeights[i + 1, j] : a;
                    // float c = (j + 1 < kernelSize) ? singleWeights[i, j + 1] : a;
                    // float d = (i + 1 < kernelSize) ? (j + 1 < kernelSize) ? singleWeights[i + 1, j + 1] : b : c;
                    // float combinedWeight = a + b + c + d;

                    //_kernelWeights[sampleIndex] = 1.0f / (float)expectedKernelSize;// singleWeights[i, j];
                   // _kernelSamples1[sampleIndex] = pixelSize * new Vector2(i - blurSize, j - blurSize);
                    sampleIndex++;
               // }
            }

            _weightsLUT = _kernelWeights.AsTexture();
            _gaussianBlurShader.SetInt("kernel_sample_count", _kernelSampleCount);
            _gaussianBlurShader.SetFloats("kernel_weights", _kernelWeights);
            //_gaussianBlurShader.SetVectorArray("kernel_sample_points", _kernelSamples1);
        }

        foregroundCloudMat.SetMatrix(_foregroundSimulationUVTransformId, _simulationUVTransform);
        if(blurSize == 0) {
            foregroundCloudMat.SetTexture(_foregroundSimulationTexId, _simulation.SimulationOutputHDR);
            foregroundCloudMat.SetInteger(_foregroundSimuilationLodId, foregroundSimulationLOD);
        } else {
            foregroundCloudMat.SetTexture(_foregroundSimulationTexId, _foregroundSimulationTex);
            foregroundCloudMat.SetInteger(_foregroundSimuilationLodId, 0);
        }
    }
}
