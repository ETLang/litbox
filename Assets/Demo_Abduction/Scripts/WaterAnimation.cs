using UnityEngine;

[RequireComponent(typeof(ProceduralHill))]
public class WaterAnimation : LitboxComponent
{
    [SerializeField] float rate1;
    [SerializeField] float rate2;

    ProceduralHill _waterSurface;

    private void Start()
    {
        _waterSurface = GetComponent<ProceduralHill>();
    }

    void Apply(ref float offset, float rate)
    {
        var delta = rate * Time.deltaTime;
        var next = offset + delta;
        offset = next;
    }

    protected override void Update()
    {
        base.Update();
        Apply(ref _waterSurface.layers[0].textureOffset, rate1);
        Apply(ref _waterSurface.layers[1].textureOffset, rate2);
    }
}