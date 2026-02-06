using System.Data;
using System.Linq;
using UnityEngine;

[ExecuteInEditMode]
public class HillGroupController : LitboxComponent
{
    [SerializeField] Color leftAmbience;
    [SerializeField] Color rightAmbience;
    [SerializeField] Color haze;
    [SerializeField] Color specularFilter;
    [SerializeField] float rayTracingVerticalOffset = -0.1f;

    ProceduralHill[] _hills;

    private void Start()
    {
        DetectChanges(() => leftAmbience);
        DetectChanges(() => rightAmbience);
        DetectChanges(() => specularFilter);
        DetectChanges(() => haze);
        DetectChanges(() => rayTracingVerticalOffset);

        OnEnable();
    }

    private void OnEnable()
    {
        OnInvalidated(null);
    }

    protected override void OnDisable()
    {
        base.OnDisable();

        _hills = null;
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        if(_hills == null && enabled) {
            _hills = this.GetComponentsInDescendants<ProceduralHill>().ToArray();
        }

        if (_hills != null) {
            foreach (var hill in _hills) {  
                hill.leftAmbience = leftAmbience;
                hill.rightAmbience = rightAmbience;
                hill.specularFilter = specularFilter;
                hill.haze = haze;
                hill.rayTracingVerticalOffset = rayTracingVerticalOffset;
            }
        }
    }
}