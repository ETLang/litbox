using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class SimulationCamera : MonoBehaviour {

    private ComputeShader _computeShader;

    public PhotonerGBuffer GBuffer { get; set; }

    public float VarianceEpsilon {
        get => _varianceEpsilon;
        set {
            if(_varianceEpsilon != value) {
                if(_postRenderCommands != null) {
                    _postRenderCommands.Dispose();
                    _postRenderCommands = null;
                }
            }
            _varianceEpsilon = value;
        }
    }
    private float _varianceEpsilon;

    public Texture2D TestTexture;

    private CommandBuffer _preRenderCommands;
    private CommandBuffer _postRenderCommands;
    private Camera _cam;

    public void Initialize(Transform parent, int layers) {
        _preRenderCommands = new CommandBuffer();
        _preRenderCommands.name = "Simulation Global Properties";
        _preRenderCommands.SetGlobalInt(Shader.PropertyToID("_isRayTracing"), 1);

        _cam = GetComponent<Camera>();
        _cam.transform.parent = parent;
        _cam.transform.localScale = new Vector3(1,1,1);
        _cam.transform.localRotation = Quaternion.identity;
        _cam.transform.localPosition = Vector3.zero;
        _cam.orthographic = true;
        _cam.orthographicSize = parent.lossyScale.y / 2;
        _cam.nearClipPlane = -100;
        _cam.farClipPlane = 1000;
        _cam.cullingMask = layers;
        _cam.clearFlags = CameraClearFlags.Nothing;
        _cam.allowHDR = false;
        _cam.allowMSAA = false;
        _cam.useOcclusionCulling = false;

        _cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, _preRenderCommands);
        
        gameObject.SetActive(false);

        _computeShader = (ComputeShader)Resources.Load("GBuffer");
    }

    public void Render() {
        _cam.aspect = transform.parent.lossyScale.x / transform.parent.lossyScale.y;
        _cam.orthographicSize = _cam.transform.parent.lossyScale.y / 2;
        _cam.Render();
    }

    public void ClearTargets() {
        _cam.targetTexture = null;
    }

    void OnPreRender() {
        var gBuffer = new RenderBuffer[]
        {
            GBuffer.AlbedoAlpha.colorBuffer,
            GBuffer.Transmissibility.colorBuffer,
            GBuffer.NormalRoughness.colorBuffer
        };

        RenderTexture rt = RenderTexture.active;
        RenderTexture.active = GBuffer.AlbedoAlpha;
        GL.Clear(true, true, new Color(0,0,0,1));
        RenderTexture.active = GBuffer.Transmissibility;
        GL.Clear(false, true, new Color(1,1,0,1));
        RenderTexture.active = GBuffer.NormalRoughness;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = GBuffer.QuadTreeLeaves;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = rt;
        _cam.SetTargetBuffers(gBuffer, GBuffer.AlbedoAlpha.depthBuffer);
    }

    void OnPostRender() {
        if(_postRenderCommands == null) {
           _postRenderCommands = new CommandBuffer();

            var generateGBufferMipsKernel = _computeShader.FindKernel("GenerateGBufferMips");
            int mipW = GBuffer.Transmissibility.width;
            int mipH = GBuffer.Transmissibility.height;

            _postRenderCommands.SetGlobalInt(Shader.PropertyToID("_isRayTracing"), 0);

            _postRenderCommands.SetComputeVectorParam(_computeShader, 
                "g_target_size", new Vector2(GBuffer.AlbedoAlpha.width, GBuffer.AlbedoAlpha.height));
            _postRenderCommands.SetComputeIntParam(_computeShader,
                "g_lowest_lod", (int)(GBuffer.AlbedoAlpha.mipmapCount - 3));

            for(int i = 1;i < GBuffer.Transmissibility.mipmapCount;i++) {
                mipW /= 2;
                mipH /= 2;
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel, 
                    "g_destMipLevelAlbedo", GBuffer.AlbedoAlpha, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_sourceMipLevelAlbedo", GBuffer.AlbedoAlpha, i-1);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_destMipLevelTransmissibility", GBuffer.Transmissibility, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_sourceMipLevelTransmissibility", GBuffer.Transmissibility, i-1);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_destMipLevelNormalSlope", GBuffer.NormalRoughness, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,   
                    "g_sourceMipLevelNormalSlope", GBuffer.NormalRoughness, i-1);
                _postRenderCommands.DispatchCompute(_computeShader, generateGBufferMipsKernel,
                    Math.Max(1, mipW / 8), Math.Max(1, mipH / 8), 1);
            }

            mipW = GBuffer.Transmissibility.width;
            mipH = GBuffer.Transmissibility.height;
            var computeGBufferVarianceKernel = _computeShader.FindKernel("ComputeGBufferVariance");
            var eps = VarianceEpsilon;
            for(int i = 1;i < GBuffer.Transmissibility.mipmapCount;i++) {
                mipW /= 2;
                mipH /= 2;
                eps /= 2.0f;
                _postRenderCommands.SetComputeFloatParam(_computeShader, 
                    "g_TransmissibilityVariationEpsilon", eps);
                _postRenderCommands.SetComputeTextureParam(_computeShader, computeGBufferVarianceKernel,
                    "g_sourceMipLevelTransmissibility", GBuffer.Transmissibility, i);
                _postRenderCommands.DispatchCompute(_computeShader, computeGBufferVarianceKernel,
                    Math.Max(1, mipW / 8), Math.Max(1, mipH / 8), 1);
            }

            var generateQuadTreeKernel = _computeShader.FindKernel("GenerateGBufferQuadTree");
            _postRenderCommands.SetComputeTextureParam(_computeShader, generateQuadTreeKernel,
                "g_transmissibility", GBuffer.Transmissibility);
            _postRenderCommands.SetComputeTextureParam(_computeShader, generateQuadTreeKernel,
                "g_destQuadTreeLeaves", GBuffer.QuadTreeLeaves, 0);
            _postRenderCommands.DispatchCompute(_computeShader, generateQuadTreeKernel,
                Math.Max(1, GBuffer.QuadTreeLeaves.width / 8), Math.Max(1, GBuffer.QuadTreeLeaves.height / 8), 1);

        }

        Graphics.ExecuteCommandBuffer(_postRenderCommands);
    }    
}