using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

public static class Extensions {
    public static IEnumerable<Transform> GetChildren(this Component _this)
    {
        for(int i = 0;i < _this.transform.childCount;i++) {
            yield return _this.transform.GetChild(i);
        }
    }

    public static IEnumerable<T> GetComponentsInDescendants<T>(this Component _this)
    {
        if (_this.TryGetComponent<T>(out var comp)) {
            yield return comp;
        }

        foreach(var child in _this.GetChildren()) {
            foreach(var desc in GetComponentsInDescendants<T>(child)) {
                yield return desc;
            }
        }
    }
}