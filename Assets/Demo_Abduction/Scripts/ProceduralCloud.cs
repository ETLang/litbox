using UnityEngine;

public class ProceduralCloud : PhotonerComponent
{
    [SerializeField] public Texture2D cloudTexture;
    [SerializeField] public float cloudScale = 1f;
    [SerializeField] Material foregroundMat;

    private MaterialPropertyBlock _propertyBlock;
    private Renderer _renderer;

    private static int _cloudTextureId = Shader.PropertyToID("_CloudTexture");
    private static int _cloudScaleId = Shader.PropertyToID("_CloudScale");

    public ProceduralCloud()
    {
        DetectChanges(() => cloudTexture);
        DetectChanges(() => cloudScale);
    }

    private void Awake()
    {
        _renderer = GetComponent<Renderer>();
        _propertyBlock = new MaterialPropertyBlock();
    }

    protected override void Update()
    {
        base.Update();
        _renderer.GetPropertyBlock(_propertyBlock);
        _propertyBlock.SetTexture(_cloudTextureId, cloudTexture);
        _propertyBlock.SetFloat(_cloudScaleId, cloudScale);
        _renderer.SetPropertyBlock(_propertyBlock);
    }
}