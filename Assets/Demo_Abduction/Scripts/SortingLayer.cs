using UnityEngine;

public class SortingLayer : LitboxComponent
{
    [SerializeField] RTLayer layer;
    [SerializeField] int order;

    void Start()
    {
        DetectChanges(() => layer);
        DetectChanges(() => order);
        OnInvalidated(null);
    }

    protected override void OnInvalidated(string group)
    {
        var renderer = GetComponent<Renderer>();
        renderer.sortingLayerName = layer.ToString();
        renderer.sortingOrder = order;
    }
}