using System;
using System.Collections.Generic;

public interface IChangeManager
{
    bool Check();
}

public class ChangeManager<T> : IChangeManager
{
    private T _previous = default;
    private Func<T> _getter;
    private string _group;
    private Action<string> _invalidate;

    public ChangeManager(Func<T> getter, string group, Action<string> invalidate)
    {
        _getter = getter;
        _group = group;
        _invalidate = invalidate;
        _previous = getter();
    }

    public bool Check()
    {
        bool changed = false;

        var current = _getter();
        var c_null = current == null;
        var p_null = _previous == null;

        if (c_null != p_null) {
            changed = true;
        } else if (!p_null) {
            changed = !_previous.Equals(current);
        }

        if (changed) {
            _previous = current;
            _invalidate(_group);
        }

        return changed;
    }
}

public class SetChangeManager<T> : IChangeManager
{
    private bool _previousNull = false;
    private HashSet<T> _previous = new HashSet<T>();
    private Func<IList<T>> _getter;
    private string _group;
    private Action<string> _invalidate;

    void GatherPrevious(IList<T> current)
    {
        _previousNull = (current == null);
        _previous.Clear();

        if(!_previousNull) {
            foreach(var x in current)
                _previous.Add(x);
        }
    }

    public SetChangeManager(Func<IList<T>> getter, string group, Action<string> invalidate)
    {
        _getter = getter;
        _group = group;
        _invalidate = invalidate;
        GatherPrevious(_getter());
    }

    public bool Check()
    {
        bool changed = false;

        var current = _getter();
        var c_null = current == null;

        if (c_null != _previousNull) {
            changed = true;
        } else if (!_previousNull) {
            changed = !_previous.SetEquals(current);
        }

        if (changed) {
            GatherPrevious(current);
            _invalidate(_group);
        }

        return changed;
    }
}