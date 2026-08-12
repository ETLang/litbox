using UnityEngine;

public enum RTSpriteLayer
{
    Background,
    Inlay,
    Overlay
}

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class RTDemoSprite : LitboxComponent
{
    [SerializeField] public RTSpriteLayer layer;
    [SerializeField] public int sortingOrder;
    [SerializeField] public Texture texture;
    [SerializeField] public Color colorMod;
    [SerializeField, ColorUsage(false, false)] public Color lightMod;
    [SerializeField, ColorUsage(false, true)] public Color ambience = Color.white;
    [SerializeField, ColorUsage(false, true)] public Color emissive = Color.white;
    [SerializeField, Range(0,1)] public float metallic = 0;
    [SerializeField] public float lightDetail;
    [SerializeField] public bool bypassTonemapping;

    private static int _colorId = Shader.PropertyToID("_Color");
    private static int _mainTexId = Shader.PropertyToID("_MainTex");
    private static int _lightMapId = Shader.PropertyToID("_LightMap");
    private static int _metallicId = Shader.PropertyToID("_Metallic");
    private static int _ambienceId = Shader.PropertyToID("_Ambience");
    private static int _lightModId = Shader.PropertyToID("_LightMod");
    private static int _emissiveId = Shader.PropertyToID("_Emissive");
    private static int _lightDetailId = Shader.PropertyToID("_LightDetail");
    private static int _lightingUVTransformId = Shader.PropertyToID("_LightingUVTransform");


    private MeshRenderer _renderer;
    private MaterialPropertyBlock _propertyBlock;
    private Texture _lightMap;
    private Vector4 _simulationUVTransform;
    private BindSimulationToCamera _binder;
    private Simulation _sim;
    private Shader _targetShader;

    void Start()
    {
        _propertyBlock = new MaterialPropertyBlock();
        _renderer = GetComponent<MeshRenderer>();

        // Ensure the renderer uses the correct shader
        _targetShader = Shader.Find("Portfolio/PortfolioSpriteShader");
        if (_renderer.sharedMaterial == null || _renderer.sharedMaterial.shader != _targetShader)
        {
            _renderer.material = new Material(_targetShader);
        }

        DetectChanges(() => layer, "layer");
        DetectChanges(() => sortingOrder, "layer");
        DetectChanges(() => texture);
        DetectChanges(() => colorMod);
        DetectChanges(() => lightMod);
        DetectChanges(() => ambience);
        DetectChanges(() => emissive);
        DetectChanges(() => metallic);
        DetectChanges(() => lightDetail);
        DetectChanges(() => _lightMap);
        DetectChanges(() => _simulationUVTransform);
        UpdateLayerProperties();
        UpdateMaterialProperties();
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        if(group == "layer")
        {
            UpdateLayerProperties();
        }
        else
        {
            UpdateMaterialProperties();
        }
    }

    private void UpdateLayerProperties()
    {
        // When the layer is Background, ensure it renders behind the default layer.
        // This often means using the "Default" sorting layer with a negative sorting order,
        // as custom sorting layers might be ordered incorrectly in Project Settings.
        if (layer == RTSpriteLayer.Background)
        {
            _renderer.sortingLayerName = "Default";
            _renderer.sortingOrder = sortingOrder - 1000; // Subtract a large value to ensure it's far behind
        }
        else
        {
            _renderer.sortingLayerName = layer.ToString();
            _renderer.sortingOrder = sortingOrder;
        }
    }

    private void UpdateMaterialProperties()
    {
        if(_renderer != null) {
            if(texture) {
                _propertyBlock.SetTexture(_mainTexId, texture);
            } else {
                _propertyBlock.SetTexture(_mainTexId, Texture2D.whiteTexture);
            }
            if(_lightMap) {
                _propertyBlock.SetTexture(_lightMapId, _lightMap); 
            } else {
                _propertyBlock.SetTexture(_lightMapId, Texture2D.blackTexture);
            }
            _propertyBlock.SetColor(_colorId, colorMod);
            _propertyBlock.SetFloat(_metallicId, metallic);
            _propertyBlock.SetColor(_ambienceId, ambience);
            _propertyBlock.SetColor(_emissiveId, emissive);
            _propertyBlock.SetFloat(_lightDetailId, lightDetail);
            _propertyBlock.SetColor(_lightModId, lightMod);
            _propertyBlock.SetVector(_lightingUVTransformId, _simulationUVTransform);

            _renderer.SetPropertyBlock(_propertyBlock);
        }
    }

    protected override void Update()
    {
        base.Update();

        if(_binder == null) {
            _binder = Camera.main.GetComponentInChildren<BindSimulationToCamera>();
        }

        if(_binder)
        {
            _sim = _binder.GetComponent<Simulation>();
        }

        if(_sim == null)
        {
            _sim = Object.FindFirstObjectByType<Simulation>();
        }

        if(_sim)
        {        
            _lightMap = _sim.SimulationOutput;

            var m = _sim.ScreenSpaceToSimulationUV * Matrix4x4.Scale(new Vector3(1,-1,1));
            _simulationUVTransform = new Vector4(m[0,0], m[1,1], m[0,3], m[1,3]);
        }
    }
}