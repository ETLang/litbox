using UnityEngine;
using System.Runtime.InteropServices;

public class MobileDetector
{
#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern bool IsMobile();
    public static bool IsMobileDevice => IsMobile();
#else
    public static bool IsMobileDevice => false;
#endif
}