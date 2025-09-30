using System;
using UnityEngine;

public class SimulationRunner : MonoBehaviour
{
    public static Action OnRender { get; set; }

    private void OnPreRender()
    {
        OnRender?.Invoke();
    }
}