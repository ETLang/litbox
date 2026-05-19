using System.Linq;
using UnityEngine;

public class SilhouetteActor : LitboxComponent
{
    [SerializeField, Range(-3, 0)] float density = -0.1f;
    [SerializeField, Range(0, 1)] float shine = 0.2f;
    [SerializeField] Color color = Color.black;

    static int _ShineId = Shader.PropertyToID("_Shine");
    static int _lightMapId = Shader.PropertyToID("_lightMap");


    RTObject[] _tracedChildren;
    MeshRenderer[] _illuminatedChildren;
    BindSimulationToCamera _binder;
    Texture _lightMap;
    Matrix4x4 _simulationUVTransform;

    void Start()
    {
        var silhouetteShader = Shader.Find("Abductor/SilhouetteShader");
        _tracedChildren = GetComponentsInChildren<RTObject>();
        _illuminatedChildren = GetComponentsInChildren<MeshRenderer>().Where(renderer => renderer.sharedMaterial.shader == silhouetteShader).ToArray();

        DetectChanges(() => density, "material");
        DetectChanges(() => shine, "material");
        DetectChanges(() => color, "material");
        DetectChanges(() => _lightMap, "material");
        DetectChanges(() => _simulationUVTransform, "material");
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        if(group == "material")
        {
            UpdateMaterialProperties();
        }
    }

    void UpdateMaterialProperties()
    {
        foreach(var traced in _tracedChildren)
        {
            traced.substrateLogDensity = density;
        }

        foreach(var illuminated in _illuminatedChildren)
        {
            illuminated.material.color = color;
            illuminated.material.SetFloat(_ShineId, shine);
            illuminated.material.SetMatrix("_LightingUVTransform", _simulationUVTransform);
            if(_lightMap != null) {
                illuminated.material.SetTexture("_lightMap", _lightMap); 
            }
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
            _lightMap = _binder.GetComponent<Simulation>().SimulationOutput;
            _simulationUVTransform = _binder.ScreenToSimulationUVTransform;
        }
    }
}