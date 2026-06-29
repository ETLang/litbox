import './style.css';
import { RotatingCube } from './rotating_cube.ts';

// --- DOM ELEMENT SELECTION ---
const appContainer = document.querySelector('.app-container') as HTMLElement;
const activityBarButtons = document.querySelectorAll('.activity-bar button');
const sidebarPane = document.querySelector('.sidebar-pane') as HTMLElement;
const workspaceViewport = document.querySelector('.workspace-viewport') as HTMLElement;
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const resumeView = document.querySelector('.resume-view') as HTMLElement;

// --- CONSOLE LOG CAPTURE ---
const consoleContainer = document.getElementById('console-container');

if (consoleContainer) {
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;

    function appendLog(message: string, type: 'log' | 'warn' | 'error') {
        const logEntry = document.createElement('div');
        logEntry.textContent = message;
        logEntry.classList.add(`log-${type}`);
        consoleContainer.appendChild(logEntry);
        // Limit the number of messages to prevent performance issues
        if (consoleContainer.children.length > 50) {
            consoleContainer.removeChild(consoleContainer.children[0]);
        }
    }

    console.log = (...args: any[]) => {
        originalLog(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'log');
    };

    console.warn = (...args: any[]) => {
        originalWarn(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'warn');
    };

    console.error = (...args: any[]) => {
        originalError(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'error');
    };
}


// --- VIEW DATA MODEL ---
const viewContent = {
    intro: {
        sidebar: `
            <h3>Introduction</h3>
            <p>A brief bio and structural navigation hints.</p>
        `,
    },
    litbox: {
        sidebar: `
            <h3>Litbox Config</h3>
            <p>Rays/Pixel: <input type="range" min="1" max="100" value="50" class="slider"></p>
            <p>Bounce Depth: <input type="range" min="1" max="10" value="5" class="slider"></p>
        `,
    },
    fractals: {
        sidebar: `
            <h3>Fractal Parameters</h3>
            <p>Zoom: <input type="range" min="1" max="1000" value="100" class="slider"></p>
            <p>Max Iterations: <input type="range" min="10" max="1000" value="200" class="slider"></p>
        `,
    },
    about: {
        sidebar: `
            <h3>Contact</h3>
            <ul>
                <li><a href="mailto:example@example.com">Email</a></li>
                <li><a href="https://github.com" target="_blank">GitHub</a></li>
                <li><a href="https://linkedin.com" target="_blank">LinkedIn</a></li>
            </ul>
        `,
    },
};

type ViewKey = keyof typeof viewContent;

// --- VIEW SWITCHING LOGIC ---
function updateView(view: ViewKey) {
    // Update container attribute for CSS targeting
    appContainer.dataset.activeView = view;

    // Update active button state
    activityBarButtons.forEach(button => {
        button.classList.toggle('active', (button as HTMLElement).dataset.view === view);
    });

    // Update sidebar content
    sidebarPane.innerHTML = viewContent[view].sidebar;

    // Show/hide main content
    const isAboutView = view === 'about';
    resumeView.style.display = isAboutView ? 'block' : 'none';
    canvas.style.display = isAboutView ? 'none' : 'block';
}

// --- EVENT LISTENERS ---
activityBarButtons.forEach(button => {
    button.addEventListener('click', () => {
        const view = (button as HTMLElement).dataset.view;
        if (view && view in viewContent) {
            updateView(view as ViewKey);
        }
    });
});

// --- LAYOUT & RESIZE LOGIC ---
function updateLayout() {
    // To break the feedback loop, we must measure the viewport's size without
    // the influence of the canvas's aspect ratio.
    // 1. Temporarily hide the canvas so it doesn't affect the layout.
    const originalDisplay = canvas.style.display;
    canvas.style.display = 'none';

    // 2. Now, the viewport's dimensions are purely determined by the CSS grid.
    const rect = workspaceViewport.getBoundingClientRect();

    // 3. Restore the canvas's visibility.
    canvas.style.display = originalDisplay;

    // 4. Apply the correct dimensions to the canvas.
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
        if (cube) {
            // Manually trigger a render to avoid stretching during resize.
            cube.render();
        }
    }
}

// --- INITIALIZE ---
// Set default view
updateView('intro');

// Initialize WebGPU Cube
let cube: RotatingCube | null = null;
if (canvas) {
    cube = new RotatingCube(canvas);

    // Set initial size
    updateLayout();

    // Use ResizeObserver on the main container to react to any size changes
    const resizeObserver = new ResizeObserver(() => {
        updateLayout();
    });
    resizeObserver.observe(appContainer);
    
    cube.start();
} else {
    console.error("Canvas element not found!");
}
