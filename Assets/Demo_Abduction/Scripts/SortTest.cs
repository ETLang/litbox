using UnityEngine;

[ExecuteAlways]
public class SortTest : MonoBehaviour
{
    [SerializeField] RTLayer sortingLayer;
    [SerializeField] int sortOrder;
    [SerializeField] Color color;

    void Update()
    {
        var renderer = GetComponent<Renderer>();
        renderer.sortingLayerName = sortingLayer.ToString();
        renderer.sortingOrder = sortOrder;
        renderer.material.color = color;
    }
}