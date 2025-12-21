using System;

public interface IChangeManager
{
    bool Check();
}

public class ChangeManager<T> : IChangeManager
{
    private T _previous = default(T);
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