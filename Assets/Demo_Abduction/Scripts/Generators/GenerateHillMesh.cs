using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

public class GenerateHillMesh : GeneratorBase<Mesh>
{
    [SerializeField] int rows = 10;
    [SerializeField] int cols = 10;
    [SerializeField, ReadOnly] Mesh mesh;

    public override Object asset => mesh;

    GenerateHillMesh()
    {
        DetectChanges(() => rows);
        DetectChanges(() => cols);
    }

    public override void Populate(Object asset)
    {
        mesh = (Mesh)asset;
        mesh.Clear();

        var cellSizeX = 10.0f / cols;
        var cellSizeY = 10.0f / rows;

        // Ensure that the grid dimensions are at least 1x1.
        if (rows <= 0 || cols <= 0) {
            Debug.LogError("Rows and columns must be greater than 0.");
            return;
        }

        // Lists to store the mesh data before assigning it to the mesh.
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uvs = new List<Vector2>();

        // Calculate the total number of vertices needed.
        int totalVertices = (rows + 1) * (cols + 1);
        vertices.Capacity = totalVertices;
        uvs.Capacity = totalVertices;

        // Loop through the grid to create all the vertices and UVs.
        // The loops run up to and including the outer edges, which is why we use <=.
        for (int i = 0; i <= rows; i++) {
            for (int j = 0; j <= cols; j++) {
                // float w = (1 - Mathf.Cos(u * Mathf.PI)) / 2.0f;

                float xPos = (j * cellSizeX) - (cols * cellSizeX / 2f);
                float yPos = (i * cellSizeY) - (rows * cellSizeY / 2f);
                vertices.Add(new Vector3(xPos, yPos, 0f));

                // Calculate UVs, mapping the grid to a texture.
                // UVs range from 0 to 1, where (0,0) is bottom-left and (1,1) is top-right.
                uvs.Add(new Vector2((float)j / cols, (float)i / rows));
            }
        }

        // Calculate the total number of triangles needed.
        // Each quad (cell) has 2 triangles, and each triangle has 3 vertices.
        int totalTriangles = rows * cols * 6;
        triangles.Capacity = totalTriangles;

        // Loop through the grid again to create the triangles.
        // The loops run up to the inner edges, so we use <.
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int baseIndex = (i * (cols + 1)) + j;

                // Define the first triangle (bottom-left, top-left, top-right).
                // The indices refer to the vertices list.
                triangles.Add(baseIndex);
                triangles.Add(baseIndex + cols + 1);
                triangles.Add(baseIndex + cols + 2);

                // Define the second triangle (bottom-left, top-right, bottom-right).
                triangles.Add(baseIndex);
                triangles.Add(baseIndex + cols + 2);
                triangles.Add(baseIndex + 1);
            }
        }

        // Assign the calculated data to the mesh.
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uvs.ToArray();

        // Recalculate the normals to ensure the mesh interacts correctly with lighting.
        mesh.RecalculateNormals();
    }

}
