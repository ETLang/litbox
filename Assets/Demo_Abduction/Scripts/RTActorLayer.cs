using UnityEngine;

public enum RTLayer
{
    Background,
    Inlay,
    Overlay
}

public enum RTLightMaterialSource
{
    None,
    Simulation,
}

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class RTActorLayer : LitboxComponent
{
    [SerializeField] public RTLayer layer;
    [SerializeField] public int sortingOrder;
    [SerializeField] public Texture texture;
    [SerializeField] public Color color;
    [SerializeField, ColorUsage(false, true)] public Color ambience = Color.white;
    [SerializeField, Range(0,1)] public float metallic = 0;

    private static int _colorId = Shader.PropertyToID("_Color");
    private static int _mainTexId = Shader.PropertyToID("_MainTex");
    private static int _lightMapId = Shader.PropertyToID("_lightMap");
    private static int _metallicId = Shader.PropertyToID("_Metallic");
    private static int _ambienceId = Shader.PropertyToID("_Ambience");

    private MeshRenderer _renderer;
    private MaterialPropertyBlock _propertyBlock;
    private Texture _lightMap;
    private Matrix4x4 _simulationUVTransform;
    private BindSimulationToCamera _binder;

    void Start()
    {
        _propertyBlock = new MaterialPropertyBlock();
        _renderer = GetComponent<MeshRenderer>();

        DetectChanges(() => layer, "layer");
        DetectChanges(() => sortingOrder, "layer");
        DetectChanges(() => texture);
        DetectChanges(() => color);
        DetectChanges(() => ambience);
        DetectChanges(() => metallic);
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
        Debug.Log(layer.ToString());
        _renderer.sortingLayerName = layer.ToString();
        _renderer.sortingOrder = sortingOrder;
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
            }
            _propertyBlock.SetColor(_colorId, color);
            _propertyBlock.SetFloat(_metallicId, metallic);
            _propertyBlock.SetColor(_ambienceId, ambience);
            _propertyBlock.SetMatrix("_LightingUVTransform", _simulationUVTransform);
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
            _lightMap = _binder.GetComponent<Simulation>().SimulationOutputHDR;
            _simulationUVTransform = _binder.ScreenToSimulationUVTransform;
        }
    }
}