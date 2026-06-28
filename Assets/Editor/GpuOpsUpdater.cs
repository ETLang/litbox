using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

// This class is an editor script and will be automatically
// invoked by Unity whenever asset events occur.
public class GpuOpsUpdater : AssetPostprocessor
{
    // Called when assets are imported, deleted, or moved.
    private static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        // If GpuOps is the only asset that's changed, then we don't need to update the registry.
        if (importedAssets.Length == 1 && 
            deletedAssets.Length == 0 && 
            movedAssets.Length == 0 && 
            AssetDatabase.GetMainAssetTypeAtPath(importedAssets[0]) == typeof(GpuOps))
        {
            return;
        }
        
        UpdateRegistry();
    }

    private static GpuOps GetInstance()
    {
        string[] guids = AssetDatabase.FindAssets("t:GpuOps");
        if (guids.Length != 1) return null;

        string path = AssetDatabase.GUIDToAssetPath(guids[0]);
        return AssetDatabase.LoadAssetAtPath<GpuOps>(path);
    }

    private static void UpdateRegistry()
    {
        GpuOps instance = GetInstance();

        if(instance == null)
        {
            Debug.LogError("Precisely one GpuOps instance is required. Make one.");
            return;
        }

        // Use a SerializedObject to safely modify the list
        SerializedObject serializedRegistry = new SerializedObject(instance);
        SerializedProperty listProperty = serializedRegistry.FindProperty("_serializableList");

        HashSet<string> currentOps = new HashSet<string>();
        for(int i = 0;i < listProperty.arraySize;i++)
        {
            var element = listProperty.GetArrayElementAtIndex(i);
            currentOps.Add(element.FindPropertyRelative("Name").stringValue);
        }
        
        listProperty.ClearArray();

        // TODO: Remove Assets/Resources when refactor is complete
        string[] shaderGuids = AssetDatabase.FindAssets("t:ComputeShader", new[] { "Assets/GpuOps", "Assets/Resources" });

        for (int i = 0; i < shaderGuids.Length; i++)
        {
            string path = AssetDatabase.GUIDToAssetPath(shaderGuids[i]);
            ComputeShader shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
            
            listProperty.InsertArrayElementAtIndex(i);
            SerializedProperty element = listProperty.GetArrayElementAtIndex(i);
            
            // Assign the name and reference
            element.FindPropertyRelative("Name").stringValue = shader.name;
            element.FindPropertyRelative("Shader").objectReferenceValue = shader;

            if(!currentOps.Contains(shader.name))
            {
                Debug.Log($"[GpuOps] Added {shader.name} to registry");
            }
            else
            {
                currentOps.Remove(shader.name);
            }
        }

        foreach(var old in currentOps)
        {
            Debug.Log($"[GpuOps] Removed {old} from registry");
        }

        serializedRegistry.ApplyModifiedProperties();
        EditorUtility.SetDirty(instance);
        AssetDatabase.SaveAssets();
    }
}
