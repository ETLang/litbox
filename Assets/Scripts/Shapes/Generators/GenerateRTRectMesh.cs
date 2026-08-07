using UnityEngine;
using Object = UnityEngine.Object;

public class GenerateRTRectMesh : GeneratorBase<Mesh>
{
    [SerializeField, ReadOnly] Mesh mesh;

    public override Object asset => mesh;

    public override void Populate(Object asset)
    {
        mesh = (Mesh)asset;
        mesh.Clear();

        var vertices = new Vector3[] {
            new Vector3(-0.5f,  0.5f,  0),
            new Vector3(    0,     0,  0),
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3( 0.5f, 0.5f, 0),
            new Vector3(    0,    0, 0),
            new Vector3(-0.5f, 0.5f, 0),
            new Vector3( 0.5f,-0.5f, 0),
            new Vector3(    0,    0, 0),
            new Vector3( 0.5f, 0.5f, 0),
            new Vector3(-0.5f,-0.5f, 0),
            new Vector3(    0,    0, 0),
            new Vector3( 0.5f,-0.5f, 0),
        };
        var normals = new Vector3[] {
            new Vector3(-1, 0, 0),
            new Vector3(-1, 0, 0),
            new Vector3(-1, 0, 0),
            new Vector3( 0, 1, 0),
            new Vector3( 0, 1, 0),
            new Vector3( 0, 1, 0),
            new Vector3( 1, 0, 0),
            new Vector3( 1, 0, 0),
            new Vector3( 1, 0, 0),
            new Vector3( 0, -1, 0),
            new Vector3( 0, -1, 0),
            new Vector3( 0, -1, 0),
        };
        var uvs = new Vector2[] {
            new Vector2(0, 1),
            new Vector2(0.5f, 0.5f),
            new Vector2(0, 0),
            new Vector2(1, 1),
            new Vector2(0.5f, 0.5f),
            new Vector2(0, 1),
            new Vector2(1, 0),
            new Vector2(0.5f, 0.5f),
            new Vector2(1, 1),
            new Vector2(0, 0),
            new Vector2(0.5f, 0.5f),
            new Vector2(1, 0),
        };
        var indices = new int[] {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
        };

        mesh.vertices = vertices;
        mesh.normals = normals;
        mesh.uv = uvs;
        mesh.triangles = indices;
    }
}
