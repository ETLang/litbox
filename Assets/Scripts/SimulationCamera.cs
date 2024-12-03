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

    public void Initialize(Transform parent, int layers) {
        var cam = GetComponent<Camera>();
        cam.transform.parent = parent;
        cam.transform.localScale = new Vector3(1,1,1);
        cam.transform.localRotation = Quaternion.identity;
        cam.transform.localPosition = Vector3.zero;
        cam.orthographic = true;
        cam.orthographicSize = parent.localScale.x / 2;
        cam.nearClipPlane = -1;
        cam.farClipPlane = 1;
        cam.cullingMask = layers;
        cam.clearFlags = CameraClearFlags.Nothing;
        cam.allowHDR = false;
        cam.allowMSAA = false;
        cam.useOcclusionCulling = false;
        
        gameObject.SetActive(false);

        _computeShader = (ComputeShader)Resources.Load("Test_Compute");
    }

    public void Render() {
        GetComponent<Camera>().Render();
    }

    public void ClearTargets() {
        GetComponent<Camera>().targetTexture = null;
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
        GetComponent<Camera>().SetTargetBuffers(gBuffer, GBufferAlbedo.depthBuffer);
    }

    void OnPostRender() {
        if(_postRenderCommands == null) {
           _postRenderCommands = new CommandBuffer();

            var generateGBufferMipsKernel = _computeShader.FindKernel("GenerateGBufferMips");
            int mipSize = GBufferTransmissibility.width;

            _postRenderCommands.SetComputeVectorParam(_computeShader, 
                "g_target_size", new Vector2(GBufferAlbedo.width, GBufferAlbedo.height));
            _postRenderCommands.SetComputeIntParam(_computeShader,
                "g_lowest_lod", (int)(GBufferAlbedo.mipmapCount - 3));

            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipSize /= 2;
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
                    Math.Max(1, mipSize / 8), Math.Max(1, mipSize / 8), 1);
            }

            mipSize = GBufferTransmissibility.width;
            var computeGBufferVarianceKernel = _computeShader.FindKernel("ComputeGBufferVariance");
            var eps = VarianceEpsilon;
            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipSize /= 2;
                eps /= 2.0f;
                _postRenderCommands.SetComputeFloatParam(_computeShader, 
                    "g_TransmissibilityVariationEpsilon", eps);
                _postRenderCommands.SetComputeTextureParam(_computeShader, computeGBufferVarianceKernel,
                    "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i);
                _postRenderCommands.DispatchCompute(_computeShader, computeGBufferVarianceKernel,
                    Math.Max(1, mipSize / 8), Math.Max(1, mipSize / 8), 1);
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