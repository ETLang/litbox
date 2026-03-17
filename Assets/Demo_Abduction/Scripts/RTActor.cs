using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

[System.Serializable]
public struct LayerProperties
{
    public Material materialOverride;
    public Texture texture;
    public Color color;
    public Vector2 offset;
    public Vector2 scale;
}

public class RTActor : LitboxComponent
{
    [SerializeField] Mesh mesh;
    [SerializeField, Range(-5, 0)] public float density;
    [SerializeField, Range(0, 1)] public float roughness = 0;
    [SerializeField, Range(0, 1)] public float heightScale = 0;

    // [SerializeField] public LayerProperties background;
    // [SerializeField] LayerProperties rayTracing;
    // [SerializeField] LayerProperties inlay;
    // [SerializeField] LayerProperties overlay;

    [SerializeField] public RTObject rayTraced;
    [SerializeField] public RTActorLayer background;
    [SerializeField] public RTActorLayer inlay;
    [SerializeField] public RTActorLayer overlay;

    void OnValidate()
    {
        if(rayTraced == null)
        {
            rayTraced = GetComponentInChildren<RTObject>();
        }

        RTActorLayer[] layers = null;
        if(background == null || inlay == null || overlay == null)
        {
            layers = GetComponentsInChildren<RTActorLayer>();
        }

        if(background == null)
        {
            background = layers.FirstOrDefault(layer => layer.layer == RTLayer.Background);
        }

        if(inlay == null)
        {
            inlay = layers.FirstOrDefault(layer => layer.layer == RTLayer.Inlay);
        }

        if(overlay == null)
        {
            overlay = layers.FirstOrDefault(layer => layer.layer == RTLayer.Overlay);
        }
    }
}