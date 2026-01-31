using static UnityEngine.GraphicsBuffer;
using UnityEditor;
using UnityEngine;
using System.IO;
using NUnit.Framework.Internal;
using UnityEngine.UIElements;
using System;
using Object = UnityEngine.Object;

public class ProceduralGeneratorEditor : Editor
{
    private new IProceduralGenerator target => (IProceduralGenerator)base.target;
    private string folder => "Assets/Procedural";
    private string assetPath => folder + "/" + target.gameObject.name + "_" + target.assetType.Name + ".asset";
    bool loaded = false;

    public override VisualElement CreateInspectorGUI()
    {
        target.Invalidated += Target_Invalidated;
        return base.CreateInspectorGUI();
    }

    private void Target_Invalidated(IProceduralGenerator obj)
    {
        if(!loaded)
        {
            loaded = true;
            return;
        }
        
        Generate();
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if(target.CheckChanged()) {
            //Generate();
        }

        if (GUILayout.Button("Generate & Save")) {
            Generate();
            Debug.Log($"{target.assetType.Name} saved to: {assetPath}");
        }
    }

    public void Generate()
    {
        if (!AssetDatabase.IsValidFolder(folder)) {
            Directory.CreateDirectory(Application.dataPath + "/Procedural");
            AssetDatabase.Refresh();
        }

        Object asset = AssetDatabase.LoadAssetAtPath<Object>(assetPath);
        if (asset != null) {
            target.Populate(asset);
            asset.name = target.gameObject.name + "_" + asset.GetType().Name;
            EditorUtility.SetDirty(asset);
        } else {
            asset = target.CreateAsset();
            target.Populate(asset);
            asset.name = target.gameObject.name + "_" + asset.GetType().Name;
            AssetDatabase.CreateAsset(asset, assetPath);
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
    }
}

[CustomEditor(typeof(GenerateHillMesh))]
public class GenerateHillMeshEditor : ProceduralGeneratorEditor { }

[CustomEditor(typeof(GenerateGradientTexture))]
public class GenerateHillTextureEditor : ProceduralGeneratorEditor { }

[CustomEditor(typeof(GenerateCloudTexture))]
public class GenerateCloudTextureEditor : ProceduralGeneratorEditor { }