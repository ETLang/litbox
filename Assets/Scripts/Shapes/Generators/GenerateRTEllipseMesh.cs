using UnityEngine;
using Object = UnityEngine.Object;

public class GenerateRTEllipseMesh : GeneratorBase<Mesh>
{
    [SerializeField] int segmentCount = 32;
    [SerializeField, ReadOnly] Mesh mesh;

    public override Object asset => mesh;

    GenerateRTEllipseMesh()
    {
        DetectChanges(() => segmentCount);
    }

    public override void Populate(Object asset)
    {
        mesh = (Mesh)asset;
        mesh.Clear();

        var vertices = new Vector3[segmentCount + 1];
        var normals = new Vector3[segmentCount + 1];
        var uvs = new Vector2[segmentCount + 1];
        var indices = new int[segmentCount * 3];

        vertices[0] = Vector3.zero;
        normals[0] = new Vector3(0, 0, -1);
        uvs[0] = new Vector2(0.5f, 0.5f);

        for (int i = 0; i < segmentCount; i++) {
            float angle = (float)i / segmentCount * Mathf.PI * 2f;
            float x = Mathf.Cos(angle) * 0.5f;
            float y = Mathf.Sin(angle) * 0.5f;
            vertices[i + 1] = new Vector3(x, y, 0f);
            normals[i + 1] = new Vector3(x, y, 0f);
            uvs[i + 1] = new Vector2(x + 0.5f, y + 0.5f);

            indices[i * 3 + 0] = 0;
            indices[i * 3 + 2] = i + 1;
            indices[i * 3 + 1] = (i + 1) % segmentCount + 1;
        }

        mesh.vertices = vertices;
        mesh.normals = normals;
        mesh.uv = uvs;
        mesh.triangles = indices;
    }
}
