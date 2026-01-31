using System;
using UnityEditor;
using UnityEngine;

[CustomPropertyDrawer(typeof(HexIntAttribute))]
public class HexIntDrawer : PropertyDrawer
{
    public HexIntAttribute hexIntAttribute {
        get { return ((HexIntAttribute)attribute); }
    }

    public override void OnGUI(Rect position,
                               SerializedProperty property,
                               GUIContent label)
    {
        EditorGUI.BeginChangeCheck();
        string hexValue = EditorGUI.TextField(position, label,
             property.longValue.ToString(hexIntAttribute.FormatString));

        long value = 0;

        if (hexValue.StartsWith("0x")) {
            try {
                value = Convert.ToInt64(hexValue, 16);
            }
            catch (FormatException) {
                value = 0;
            }
        } else {
            bool parsed = long.TryParse(hexValue, System.Globalization.NumberStyles.HexNumber,
                                        null, out value);
            if (!parsed) {
                value = 0;
            }
        }

        if (EditorGUI.EndChangeCheck())
            property.longValue = value;
    }
}