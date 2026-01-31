using UnityEngine;

[RequireComponent(typeof(RectTransform))]
public class PlatformCycler : MonoBehaviour
{
    RectTransform _rect;

    void Start()
    {
        _rect = GetComponent<RectTransform>();
    }

    void Update()
    {
        var w = _rect.rect.width;
        var center = Camera.main.transform.position.x;

        var left = center - w / 2.0f;
        var right = center + w / 2.0f;

        for(int i = 0;i < transform.childCount;i++) {
            var child = transform.GetChild(i);
            var pos = child.position;

            while(pos.x < left) {
                pos.x += w;
            }

            while(pos.x > right) {
                pos.x -= w;
            }

            child.position = pos;
        }
    }
}