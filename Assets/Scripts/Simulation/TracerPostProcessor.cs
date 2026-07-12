using System;
using System.Reflection;
using UnityEngine;

internal delegate void D_ComputeVolatilityLevel0(DispatchSize size, TextureView _in_normal, TextureView _out_volatility);
internal delegate void D_ComputeDenoiserQuadtreeLevel0(DispatchSize size,
    TextureView _in_albedo, TextureView _in_radiance, TextureView _in_density, TextureView _in_volatility_0,
    float _albedo_luminance_threshold, float _albedo_chroma_threshold, float _volatility_threshold, float _log_density_threshold,
    TextureView _out_albedo_min, TextureView _out_albedo_max,
    // TextureView _out_radiance_min, TextureView _out_radiance_max,
    TextureView _out_density_min_max_volatility, TextureView _out_quadtree);
internal delegate void D_ComputeDenoiserQuadtree(DispatchSize size, TextureView _in_albedo_min, TextureView _in_albedo_max,
    //TextureView _in_radiance_min, TextureView _in_radiance_max,
    TextureView _in_density_min_max_volatility, TextureView _in_quadtree,
    float _albedo_luminance_threshold, float _albedo_chroma_threshold, float _volatility_threshold, float _log_density_threshold,
    TextureView _out_albedo_min, TextureView _out_albedo_max,
    // TextureView _out_radiance_min, TextureView _out_radiance_max,
    TextureView _out_density_min_max_volatility, TextureView _out_quadtree);


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

    public float AlbedoLuminanceThreshold { get; set; } = 0.1f;
    public float AlbedoChromaThreshold { get; set; } = 0.2f;
    public float VolatilityThreshold { get; set; } = 0.002f;
    public float LogDensityThreshold { get; set; } = 0.002f;

    private D_ComputeVolatilityLevel0 ComputeVolatilityLevel0;
    private D_ComputeDenoiserQuadtreeLevel0 ComputeDenoiserQuadtreeLevel0;
    private D_ComputeDenoiserQuadtree ComputeDenoiserQuadtree;

    private TracerPostProcessor()
    {
        _postProcessingShader = (ComputeShader)Resources.Load("TracerPostProcessing");
        ComputeVolatilityLevel0 = GpuOps.GetCallable<D_ComputeVolatilityLevel0>("ComputeVolatilityLevel0");
        ComputeDenoiserQuadtreeLevel0 = GpuOps.GetCallable<D_ComputeDenoiserQuadtreeLevel0>("ComputeDenoiserQuadtreeLevel0");
        ComputeDenoiserQuadtree = GpuOps.GetCallable<D_ComputeDenoiserQuadtree>("ComputeDenoiserQuadtree");

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

    public RenderTexture V0 {get; set;}
    public RenderTexture LogDensityVolatility {get; set;}
    public RenderTexture AlbedoMin {get;set;}
    public RenderTexture AlbedoMax {get;set;}
    public RenderTexture Quadtree {get;set;}


    public void GenerateDenoisingFilterQuadtree(RenderTexture albedo, RenderTexture normal, RenderTexture density, RenderTexture hdrFinal, RenderTexture destQuadtree)
    {
        var albedoMin = AlbedoMin ?? BufferManager.AcquireTexture(destQuadtree, RenderTextureFormat.ARGBHalf);
        AlbedoMin = albedoMin;
        var albedoMax = AlbedoMax ?? BufferManager.AcquireTexture(destQuadtree, RenderTextureFormat.ARGBHalf);
        AlbedoMax = albedoMax;
        var volatilityLevel0 = V0 ?? BufferManager.AcquireTexture(normal.width, normal.height, RenderTextureFormat.RFloat);
        V0 = volatilityLevel0;
        var logDensityRangeVolatility = LogDensityVolatility ?? BufferManager.AcquireTexture(destQuadtree, RenderTextureFormat.ARGBFloat);
        LogDensityVolatility = logDensityRangeVolatility;
        Quadtree = destQuadtree;
        // var radianceMin = BufferManager.AcquireTexture(destQuadtree, hdrFinal.format);
        // var radianceMax = BufferManager.AcquireTexture(destQuadtree, hdrFinal.format);

        ComputeVolatilityLevel0((normal.width, normal.height), normal, volatilityLevel0);
        ComputeDenoiserQuadtreeLevel0((destQuadtree.width, destQuadtree.height),
            albedo.SelectMip(0), hdrFinal.SelectMip(0), density.SelectMip(0), volatilityLevel0,
            AlbedoLuminanceThreshold, AlbedoChromaThreshold, VolatilityThreshold, LogDensityThreshold,
            albedoMin.SelectMip(0), albedoMax.SelectMip(0), logDensityRangeVolatility.SelectMip(0), destQuadtree.SelectMip(0));

        for(int i = 1;i < destQuadtree.mipmapCount;i++)
        {
            ComputeDenoiserQuadtree((destQuadtree.MipWidth(i), destQuadtree.MipHeight(i)),
                albedoMin.SelectMip(i-1), albedoMax.SelectMip(i-1), logDensityRangeVolatility.SelectMip(i-1), destQuadtree.SelectMip(i-1),
                AlbedoLuminanceThreshold, AlbedoChromaThreshold, VolatilityThreshold, LogDensityThreshold,
                albedoMin.SelectMip(i), albedoMax.SelectMip(i), logDensityRangeVolatility.SelectMip(i), destQuadtree.SelectMip(i));
        }

        // BufferManager.Release(ref albedoMin);
        // BufferManager.Release(ref albedoMax);
        // BufferManager.Release(ref volatilityLevel0);
        //BufferManager.Release(ref logDensityRangeVolatility);
        // BufferManager.Release(ref radianceMin);
        // BufferManager.Release(ref radianceMax);
    }
}