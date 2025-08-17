using System;
using System.Collections.Generic;
using UnityEngine;

public class PhotonerDemoComponent : MonoBehaviour
{
    private List<IChangeManager> _changeManagers = new List<IChangeManager>();
    private bool _isDirty;
    private bool _autoUpdate;

    private static Action _onGlobalChangeCheck;
    public static void CheckForGlobalChanges()
    {
        _onGlobalChangeCheck?.Invoke();
    }

    public PhotonerDemoComponent(bool autoUpdate = true)
    {
        _autoUpdate = autoUpdate;

        _onGlobalChangeCheck += () => OnInvalidated();
    }

    private void Invalidate()
    {
        _isDirty = true;
    }

    protected virtual void OnInvalidated()
    {
    }

    protected void DetectChanges<PT>(Func<PT> getter)
    {
        _changeManagers.Add(new ChangeManager<PT>(getter, Invalidate));
    }

    public bool CheckChanged()
    {
        foreach (var cm in _changeManagers) {
            cm.Check();
        }

        var changed = _isDirty;
        if (_isDirty) {
            _isDirty = false;
            OnInvalidated();
        }

        return changed;
    }

    protected virtual void Update()
    {
        if (_autoUpdate) {
            CheckChanged();
        }
    }
}
