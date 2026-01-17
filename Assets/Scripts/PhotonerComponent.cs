using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PhotonerComponent : DisposalHelperComponent
{
    private List<IChangeManager> _changeManagers = new List<IChangeManager>();
    private List<string> _dirtyGroups = new List<string>();
    private bool _autoUpdate;

    private static Action<string> _onGlobalChangeCheck;

    #if UNITY_EDITOR
    private bool IsAvailable => (bool)this && UnityEditor.EditorApplication.isPlaying;
    #else
    private bool IsAvailable => (bool)this;
    #endif

    public static void CheckForGlobalChanges()
    {
        _onGlobalChangeCheck?.Invoke(null);
    }

    public PhotonerComponent(bool autoUpdate = true)
    {
        _autoUpdate = autoUpdate;

        _onGlobalChangeCheck += OnGlobalChangeCheck;
    }

    private void Invalidate(string group)
    {
        _dirtyGroups.Add(group);
    }

    private bool IsInActiveScene => gameObject.scene.IsValid() && gameObject.scene.isLoaded;

    protected void OnGlobalChangeCheck(string group)
    {
        if(!IsAvailable) return;
        OnInvalidated(null);
    }

    protected virtual void OnInvalidated(string group)
    {
    }

    protected void DetectChanges<PT>(Func<PT> getter, string group=null)
    {
        _changeManagers.Add(new ChangeManager<PT>(getter, group, Invalidate));
    }

    protected void DetectSetChanges<E>(Func<IList<E>> getter, string group=null)
    {
        _changeManagers.Add(new SetChangeManager<E>(getter, group, Invalidate));
    }

    public bool CheckChanged()
    {
        foreach (var cm in _changeManagers) {
            cm.Check();
        }

        var changed = _dirtyGroups.Count > 0;
        if (changed) {
            var groups = _dirtyGroups.ToList();
            _dirtyGroups.Clear();
            
            if(IsAvailable) {
                foreach (var group in groups) {
                    OnInvalidated(group);
                }
            }
        }

        return changed;
    }

    protected virtual void Update()
    {
        if (_autoUpdate) {
            CheckChanged();
        }
    }

    protected virtual void OnDestroy()
    {
        _onGlobalChangeCheck -= OnInvalidated;
    }
}
