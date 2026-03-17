using System;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(RTActor))]
public class RTActorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var actor = (RTActor)target;
        serializedObject.Update();

        EditorGUILayout.LabelField("Actor Dashboard", EditorStyles.boldLabel);
        EditorGUILayout.Space();

        Mesh canonicalMesh = null;

        if(actor.rayTraced)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Raytracing Settings (" + actor.rayTraced.name + ")", EditorStyles.miniBoldLabel);

            var meshFilter = actor.rayTraced.GetComponent<MeshFilter>();
            if(meshFilter != null)
            {
                meshFilter.sharedMesh = (Mesh)EditorGUILayout.ObjectField("Mesh", meshFilter.sharedMesh, typeof(Mesh), meshFilter);
                canonicalMesh = meshFilter.sharedMesh;
            }

            actor.rayTraced.texture = (Texture)EditorGUILayout.ObjectField("Texture", actor.rayTraced.texture, typeof(Texture), actor.rayTraced);
            actor.rayTraced.normal = (Texture)EditorGUILayout.ObjectField("Normal", actor.rayTraced.normal, typeof(Texture), actor.rayTraced);
            actor.rayTraced.color = EditorGUILayout.ColorField("Color", actor.rayTraced.color);
            actor.rayTraced.substrateLogDensity = EditorGUILayout.Slider("Density", actor.rayTraced.substrateLogDensity, -5, 0);
            actor.rayTraced.particleAlignment = 1-EditorGUILayout.Slider("Rougness", 1-actor.rayTraced.particleAlignment, 0, 1);
            actor.rayTraced.heightScale = EditorGUILayout.Slider("Height Scale", actor.rayTraced.heightScale, 0, 1);

            EditorGUILayout.EndVertical();
        }

        if(actor.background)
        {
            var meshFilter = actor.background.GetComponent<MeshFilter>();
            if(meshFilter != null && meshFilter.sharedMesh == null)
            {
                meshFilter.sharedMesh = canonicalMesh;
            }

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Background Settings (" + actor.background.name + ")", EditorStyles.miniBoldLabel);
            actor.background.ambience = EditorGUILayout.ColorField("Ambience", actor.background.ambience);
            actor.background.color = EditorGUILayout.ColorField("Color", actor.background.color);
            actor.background.metallic = EditorGUILayout.Slider("Metallic", actor.background.metallic, 0, 1);
            actor.background.texture = (Texture)EditorGUILayout.ObjectField("Texture", actor.background.texture, typeof(Texture), actor.background, GUILayout.Height(30));
            EditorGUILayout.EndVertical();
        }

        if(actor.inlay)
        {
            var meshFilter = actor.inlay.GetComponent<MeshFilter>();
            if(meshFilter != null && meshFilter.sharedMesh == null)
            {
                meshFilter.sharedMesh = canonicalMesh;
            }

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Inlay Settings (" + actor.inlay.name + ")", EditorStyles.miniBoldLabel);
            actor.inlay.ambience = EditorGUILayout.ColorField("Ambience", actor.inlay.ambience);
            actor.inlay.color = EditorGUILayout.ColorField("Color", actor.inlay.color);
            actor.inlay.metallic = EditorGUILayout.Slider("Metallic", actor.inlay.metallic, 0, 1);
            actor.inlay.texture = (Texture)EditorGUILayout.ObjectField("Texture", actor.inlay.texture, typeof(Texture), actor.inlay, GUILayout.Height(30));
            EditorGUILayout.EndVertical();
        }

        if(actor.overlay)
        {
            var meshFilter = actor.overlay.GetComponent<MeshFilter>();
            if(meshFilter != null && meshFilter.sharedMesh == null)
            {
                meshFilter.sharedMesh = canonicalMesh;
            }

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("Overlay Settings (" + actor.overlay.name + ")", EditorStyles.miniBoldLabel);
            actor.overlay.ambience = EditorGUILayout.ColorField("Ambience", actor.overlay.ambience);
            actor.overlay.color = EditorGUILayout.ColorField("Color", actor.overlay.color);
            actor.overlay.metallic = EditorGUILayout.Slider("Metallic", actor.overlay.metallic, 0, 1);
            actor.overlay.texture = (Texture)EditorGUILayout.ObjectField("Texture", actor.overlay.texture, typeof(Texture), actor.overlay, GUILayout.Height(30));
            EditorGUILayout.EndVertical();
        }
    }
}