using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class DiagnosticHelpers
{
    public static void Visualize(this IEnumerable<float> data, Texture2D target, bool normalize=true) {
        var size = target.width * target.height;

        var min = data.Min();
        var max = data.Max();
        var med = data.OrderBy(k => k).ElementAt(size / 2);
        var um = (med - min) / (max - min);
        var p = 1;//Mathf.Log(0.5f) / Mathf.Log(um);

        if(normalize) {        
            target.SetPixels(data.Select(x => {
                var u = Mathf.Pow((x - min) / (max - min), p);
                return new Color(u,u,u,1);
            }).ToArray());
        } else {
            target.SetPixels(data.Select(x => new Color(x,x,x,1)).ToArray());
        }

        target.Apply();
    }
}