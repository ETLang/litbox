using System;
using System.Reflection;
using UnityEngine;

public class TracerPostProcessor : Disposable
{
    private ComputeShader _postProcessingShader;
    private int[] _computeCVAndMipsKernel;
    private int[] _generateMipsKernel;
    private int _filterVarianceKernel;

    private static TracerPostProcessor _Instance;
    public static TracerPostProcessor Instance =>
        _Instance ?? (_Instance = new TracerPostProcessor());

    private static int _SourceAId = Shader.PropertyToID("_sourceA");
    private static int _SourceBId = Shader.PropertyToID("_sourceB");
    private static int _OutVarianceId = Shader.PropertyToID("_out_variance");
    private static int[] _OutMipId = new int[]
    {
        Shader.PropertyToID("_out_mip0"),
        Shader.PropertyToID("_out_mip1"),
        Shader.PropertyToID("_out_mip2"),
        Shader.PropertyToID("_out_mip3"),
        Shader.PropertyToID("_out_mip4")
    };

    private static int _UnfilteredVarianceId = Shader.PropertyToID("_in_unfiltered_variance");
    private static int _AlbedoId = Shader.PropertyToID("_in_albedo");
   // private static int _NormalRoughnessId = Shader.PropertyToID("_in_normal_roughness");
    private static int _HdrFinalId = Shader.PropertyToID("_in_hdr_final");

    private static int _SigmaSpatialId = Shader.PropertyToID("_sigma_spatial");
    private static int _SigmaAlbedoId = Shader.PropertyToID("_sigma_albedo");
    private static int _SigmaLuminanceTightId = Shader.PropertyToID("_sigma_luminance_tight");
    private static int _SigmaLuminanceLooseId = Shader.PropertyToID("_sigma_luminance_loose");
    private static int _KLuminanceId = Shader.PropertyToID("_k_luminance");

    public float SigmaSpatial { get; set; } = 1.2f;
    public float SigmaAlbedo { get; set; } = 0.05f;
    public float SigmaLuminanceTight { get; set; } = 0.05f;
    public float SigmaLuminanceLoose { get; set; } = 2.5f;
    public float KLuminance { get; set; } = 2.0f;

    private TracerPostProcessor()
    {
        _postProcessingShader = (ComputeShader)Resources.Load("TracerPostProcessing");

        _computeCVAndMipsKernel = new int[]
        {
            _postProcessingShader.FindKernel("ComputeVarianceAndOneMipFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeVarianceAndTwoMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeVarianceAndThreeMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeVarianceAndFourMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeVarianceAndFiveMipsFromSamplePair"),
        };

        _generateMipsKernel = new int[]
        {
            _postProcessingShader.FindKernel("GenerateOneMip"),
            _postProcessingShader.FindKernel("GenerateTwoMips"),
            _postProcessingShader.FindKernel("GenerateThreeMips"),
            _postProcessingShader.FindKernel("GenerateFourMips"),
        };

        _filterVarianceKernel = _postProcessingShader.FindKernel("FilterVariance");
    }

    protected override void OnDispose()
    {
        _Instance = null;
        base.OnDispose();
    }

    public void ComputeVarianceAndMips(RenderTexture sourceA, RenderTexture sourceB, RenderTexture destMean, RenderTexture destVariance)
    {
        int totalMips = destMean.mipmapCount;

        int firstDispatchMipCount = Mathf.Min(totalMips, 5);
        int firstDispatchKernel = _computeCVAndMipsKernel[firstDispatchMipCount - 1];
        
        _postProcessingShader.SetTexture(firstDispatchKernel, _SourceAId, sourceA);
        _postProcessingShader.SetTexture(firstDispatchKernel, _SourceBId, sourceB);
        _postProcessingShader.SetTexture(firstDispatchKernel, _OutVarianceId, destVariance);

        for(int i = 0;i < firstDispatchMipCount;i++) {
            _postProcessingShader.SetTexture(firstDispatchKernel, _OutMipId[i], destMean, i);
        }
        _postProcessingShader.Dispatch(firstDispatchKernel, (destMean.width - 1) / 16 + 1, (destMean.height - 1) / 16 + 1, 1);

        GenerateMips(destMean, firstDispatchMipCount - 1);
    }

    public void GenerateMips(RenderTexture texture, int detailLevel = 0, int mipCount = 0)
    {
        if(mipCount == 0)
        {
            mipCount = texture.mipmapCount - 1 - detailLevel;
        }

        int lastMip = detailLevel + mipCount;
        while(detailLevel < lastMip)
        {
            int nextDispatchMipCount = Mathf.Min(lastMip - detailLevel, 4);
            int nextDispatchKernel = _generateMipsKernel[nextDispatchMipCount - 1];

            _postProcessingShader.SetTexture(nextDispatchKernel, _SourceAId, texture, detailLevel);

            for(int i = 1;i <= nextDispatchMipCount;i++) {
                _postProcessingShader.SetTexture(nextDispatchKernel, _OutMipId[i], texture, detailLevel + i);
            }
            _postProcessingShader.Dispatch(nextDispatchKernel, (texture.MipWidth(detailLevel) - 1) / 16 + 1, (texture.MipHeight(detailLevel) - 1) / 16 + 1, 1);

            detailLevel += nextDispatchMipCount;
        }
    }

    public void FilterVariance(RenderTexture source, RenderTexture dest, RenderTexture albedo, RenderTexture hdrFinal)
    {
        const int mipLevel = 2; // Variance is computed on 4x4 blocks

        _postProcessingShader.SetTexture(_filterVarianceKernel, _UnfilteredVarianceId, source);
        _postProcessingShader.SetTexture(_filterVarianceKernel, _AlbedoId, albedo, mipLevel);
        _postProcessingShader.SetTexture(_filterVarianceKernel, _HdrFinalId, hdrFinal, mipLevel);
        _postProcessingShader.SetTexture(_filterVarianceKernel, _OutVarianceId, dest);
        _postProcessingShader.SetFloat(_SigmaSpatialId, SigmaSpatial);
        _postProcessingShader.SetFloat(_SigmaAlbedoId, SigmaAlbedo);
        _postProcessingShader.SetFloat(_SigmaLuminanceTightId, SigmaLuminanceTight);
        _postProcessingShader.SetFloat(_SigmaLuminanceLooseId, SigmaLuminanceLoose);
        _postProcessingShader.SetFloat(_KLuminanceId, KLuminance);

        _postProcessingShader.Dispatch(_filterVarianceKernel, (source.width - 1) / 16 + 1, (source.height - 1) / 16 + 1, 1);
    }
}