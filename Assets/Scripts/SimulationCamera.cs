using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class SimulationCamera : MonoBehaviour {

    private ComputeShader _computeShader;

    public RenderTexture GBufferAlbedo { get; set; }
    public RenderTexture GBufferTransmissibility { get; set; }
    public RenderTexture GBufferNormalSlope { get; set; }
    public RenderTexture GBufferQuadTreeLeaves { get; set; }
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

    private CommandBuffer _postRenderCommands;
    private Camera _cam;

    public void Initialize(Transform parent, int layers) {
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
        
        gameObject.SetActive(false);

        _computeShader = (ComputeShader)Resources.Load("Simulation");
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
            GBufferAlbedo.colorBuffer,
            GBufferTransmissibility.colorBuffer,
            GBufferNormalSlope.colorBuffer
        };

        RenderTexture rt = RenderTexture.active;
        RenderTexture.active = GBufferAlbedo;
        GL.Clear(false, true, new Color(0,0,0,1));
        RenderTexture.active = GBufferTransmissibility;
        GL.Clear(false, true, new Color(1,1,0,1));
        RenderTexture.active = GBufferNormalSlope;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = GBufferQuadTreeLeaves;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = rt;
        _cam.SetTargetBuffers(gBuffer, GBufferAlbedo.depthBuffer);
    }

    void OnPostRender() {
        if(_postRenderCommands == null) {
           _postRenderCommands = new CommandBuffer();

            var generateGBufferMipsKernel = _computeShader.FindKernel("GenerateGBufferMips");
            int mipW = GBufferTransmissibility.width;
            int mipH = GBufferTransmissibility.height;

            _postRenderCommands.SetComputeVectorParam(_computeShader, 
                "g_target_size", new Vector2(GBufferAlbedo.width, GBufferAlbedo.height));
            _postRenderCommands.SetComputeIntParam(_computeShader,
                "g_lowest_lod", (int)(GBufferAlbedo.mipmapCount - 3));

            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipW /= 2;
                mipH /= 2;
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel, 
                    "g_destMipLevelAlbedo", GBufferAlbedo, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_sourceMipLevelAlbedo", GBufferAlbedo, i-1);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_destMipLevelTransmissibility", GBufferTransmissibility, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i-1);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,
                    "g_destMipLevelNormalSlope", GBufferNormalSlope, i);
                _postRenderCommands.SetComputeTextureParam(_computeShader, generateGBufferMipsKernel,   
                    "g_sourceMipLevelNormalSlope", GBufferNormalSlope, i-1);
                _postRenderCommands.DispatchCompute(_computeShader, generateGBufferMipsKernel,
                    Math.Max(1, mipW / 8), Math.Max(1, mipH / 8), 1);
            }

            mipW = GBufferTransmissibility.width;
            mipH = GBufferTransmissibility.height;
            var computeGBufferVarianceKernel = _computeShader.FindKernel("ComputeGBufferVariance");
            var eps = VarianceEpsilon;
            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipW /= 2;
                mipH /= 2;
                eps /= 2.0f;
                _postRenderCommands.SetComputeFloatParam(_computeShader, 
                    "g_TransmissibilityVariationEpsilon", eps);
                _postRenderCommands.SetComputeTextureParam(_computeShader, computeGBufferVarianceKernel,
                    "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i);
                _postRenderCommands.DispatchCompute(_computeShader, computeGBufferVarianceKernel,
                    Math.Max(1, mipW / 8), Math.Max(1, mipH / 8), 1);
            }

            var generateQuadTreeKernel = _computeShader.FindKernel("GenerateGBufferQuadTree");
            _postRenderCommands.SetComputeTextureParam(_computeShader, generateQuadTreeKernel,
                "g_transmissibility", GBufferTransmissibility);
            _postRenderCommands.SetComputeTextureParam(_computeShader, generateQuadTreeKernel,
                "g_destQuadTreeLeaves", GBufferQuadTreeLeaves, 0);
            _postRenderCommands.DispatchCompute(_computeShader, generateQuadTreeKernel,
                Math.Max(1, GBufferQuadTreeLeaves.width / 8), Math.Max(1, GBufferQuadTreeLeaves.height / 8), 1);

        }

        Graphics.ExecuteCommandBuffer(_postRenderCommands);
    }    
}