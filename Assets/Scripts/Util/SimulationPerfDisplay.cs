using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

[RequireComponent(typeof(UIDocument))]
public class SimulationPerfDisplay : MonoBehaviour
{
    public enum PerfDisplayType {
        TraversalsPerSecond,
        ConvergenceValue,
        ConvergenceTime
    }

    [SerializeField] private Simulation simulation;
    [SerializeField] private PerfDisplayType displayData;
    [SerializeField] private string labelName;

    private Label target;

    void Start()
    {
        if(!simulation)
            Debug.Log("simulation property is not set on SimulationPerfDisplay");

        // Get the root visual element from the UIDocument
        UIDocument uiDocument = GetComponent<UIDocument>();
        VisualElement root = uiDocument.rootVisualElement;

        // Find a specific UI element by its name
        target = root.Q<Label>(labelName);

        // Change the text of the label
    }

    // Update is called once per frame
    void Update()
    {
        string value = "";
        bool doUpdate = true;
        switch(displayData) {
        case PerfDisplayType.TraversalsPerSecond:
            doUpdate = !simulation.hasConverged;
            value = (simulation.TraversalsPerSecond / 1000000.0f).ToString("0.0") + " MTPS";
            break;
        case PerfDisplayType.ConvergenceValue:
            value = simulation.Convergence.ToString() + " ξ";
            break;
        case PerfDisplayType.ConvergenceTime:
            doUpdate = !simulation.hasConverged;
            value = (Time.time - simulation.ConvergenceStartTime).ToString("0.0") + "s";
            break;
        }

        if(simulation && doUpdate) {
            if(target != null) target.text = value;
            //GetComponent<TMP_Text>().text = value;
        }
    }
}
