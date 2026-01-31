using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;

public class ImportanceMap : Disposable
{
    private ComputeShader _importanceMapShader;
    private int _generateImportanceMapKernel;

    private static int _forwardRadianceAId = Shader.PropertyToID("_forwardRadianceA");
    private static int _forwardRadianceBId = Shader.PropertyToID("_forwardRadianceB");    
    private static int[] _outImportanceIds = new int[] {
        Shader.PropertyToID("_out_importance0"),
        Shader.PropertyToID("_out_importance1"),
        Shader.PropertyToID("_out_importance2"),
        Shader.PropertyToID("_out_importance3"),
    };

    public RenderTexture Map { get; private set; }

    public ImportanceMap()
    {
        _importanceMapShader = (ComputeShader)Resources.Load("ImportanceMap");
        _generateImportanceMapKernel = _importanceMapShader.FindKernel("GenerateImportanceMap");
    }

    public void Generate(RenderTexture radianceA, RenderTexture radianceB)
    {
        if (Map == null || 
            Map.width != radianceA.width / 2 || 
            Map.height != radianceA.height / 2)
        {
            if(Map != null)
            {
                GameObject.DestroyImmediate(Map);
            }

            Map = this.CreateRWTextureWithMips(radianceA.width / 2, radianceA.height / 2, RenderTextureFormat.RFloat);
        }

        _importanceMapShader.SetTexture(_generateImportanceMapKernel, _forwardRadianceAId, radianceA);
        _importanceMapShader.SetTexture(_generateImportanceMapKernel, _forwardRadianceBId, radianceB);
        for(int i = 0;i < 4;i++)
        {
            _importanceMapShader.SetTexture(_generateImportanceMapKernel, _outImportanceIds[i], Map, i);
        }
        _importanceMapShader.DispatchAutoGroup(_generateImportanceMapKernel, Map.width, Map.height, 1);
    }

    protected override void OnDispose()
    {
        base.OnDispose();

        if (Map != null) {
            GameObject.DestroyImmediate(Map);
        }
    }
}