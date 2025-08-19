using System;
using System.Collections.Generic;
using UnityEngine;

public enum GameStates
{
    Title,
    Playing,
    Paused
}

public class GameStateController : MonoBehaviour
{
    public static GameStateController Instance { get; private set; }

    [SerializeField, ReadOnly] GameStates state;
    [SerializeField] GameObject titleScreen;
    [SerializeField] GameObject pauseScreen;
    [SerializeField] Rigidbody2D player;

    Dictionary<(GameStates, GameStates), Action> _transitions = new Dictionary<(GameStates, GameStates), Action>();

    public event Action<GameStates, GameStates> StateChanged;

    public GameStates State {
        get { return state; }
        set { Transition(value); }
    }

    private void Start()
    {
        Instance = this;

        // Add transitions
        _transitions[(GameStates.Title, GameStates.Playing)] = () => {
            titleScreen.SetActive(false);
            player.WakeUp();
        };

        _transitions[(GameStates.Playing, GameStates.Paused)] = () => {
            player.Sleep();
            pauseScreen.SetActive(true);
        };

        _transitions[(GameStates.Paused, GameStates.Playing)] = () => {
            pauseScreen.SetActive(false);
            player.WakeUp();
        };

        _transitions[(GameStates.Paused, GameStates.Title)] = () => {
            pauseScreen.SetActive(false); 
            titleScreen.SetActive(true);
        };
    }

    public void PlayButtonPressed()
    {
        Transition(GameStates.Playing);
    }

    private void Transition(GameStates newState)
    {
        if(!_transitions.TryGetValue((state, newState), out var transition)) {
            throw new Exception($"Invalid transition from {state} to {newState}");
        }

        transition();
        var oldState = state;
        state = newState;
        StateChanged?.Invoke(oldState, state);
    }
}