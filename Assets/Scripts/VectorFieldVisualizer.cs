//Attach this script to a GameObject with a Camera component

using System;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Vector4 = UnityEngine.Vector4;

public enum VectorFieldType {
    EfficiencyGradient,
    RelaxedEfficiencyGradient,
    ImportanceSamplingTarget
}

public class VectorFieldVisualizer : MonoBehaviour
{
    // Draws a line from "startVertex" var to the curent mouse position.
    [SerializeField] private Material mat;
    [SerializeField] private Simulation sim;
    [SerializeField] private VectorFieldType diagnostic;
    [SerializeField] private bool cleanMagnitudes;
    [SerializeField] private bool normalize;
    [SerializeField] private bool logScale;

    void Start()
    {
    }

    void Update()
    {
    }

    void OnPostRender()
    {
        var world = sim.transform.localToWorldMatrix;

        Action<Vector2,Vector2> drawLine = (Vector2 start, Vector2 end) =>
        {
            var a = world * new Vector4(start.x, start.y, 0, 1);
            var b = world * new Vector4(end.x, end.y, 0, 1);
            a.z = -1;
            b.z = -1;
            GL.Vertex(a);
            GL.Vertex(b);
        };

        Action<Vector2[,]> standardVis = data => {
            var w = data.GetLength(0);
            var h = data.GetLength(1);
            var wStep = 1.0f / w;
            var hStep = 1.0f / h;

            var min = data.Flat().Select(v => v.magnitude).Min();
            var max = data.Flat().Select(v => v.magnitude).Max();

            if(logScale) {
                max = Mathf.Log(max + 1, 2);
            }
            
            for(int i = 0;i < w;i++) {
                for(int j = 0;j < h;j++) {
                    var pivot = new Vector2((i + 0.5f) * wStep - 0.5f, (j + 0.5f) * hStep - 0.5f);
                    var datum = data[i,j];
                    var mag = datum.magnitude;

                    if(logScale && mag != 0) {
                        datum *= Mathf.Log(mag + 1, 2) / mag;
                    }

                    if(cleanMagnitudes && mag > 1) {
                        datum.Normalize();
                    } else if(normalize) {
                        datum /= max;
                    }

                    var end = pivot + new Vector2(wStep * datum.x, hStep * datum.y) * 0.9f;
                    drawLine(pivot,end);
                }
            }        };

        mat.SetPass(0);
        GL.Begin(GL.LINES);

        switch(diagnostic) {
        case VectorFieldType.EfficiencyGradient: {
            standardVis(sim.EfficiencyGradient);
        }
        break;
        case VectorFieldType.RelaxedEfficiencyGradient: {
            standardVis(sim.RelaxedEfficiencyGradient);
        }
        break;
        case VectorFieldType.ImportanceSamplingTarget: {
            float markSize = 0.05f;
            var x = sim.ImportanceSamplingTarget - new Vector2(0.5f,0.5f);

            drawLine(new Vector2(x.x - markSize, x.y - markSize), new Vector2(x.x + markSize, x.y + markSize));
            drawLine(new Vector2(x.x + markSize, x.y - markSize), new Vector2(x.x - markSize, x.y + markSize));
        }
        break;
        }

        GL.End();
    }
}