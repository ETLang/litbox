import './style.css';
import { RotatingCube } from './rotating_cube.ts';

// --- DOM ELEMENT SELECTION ---
const appContainer = document.querySelector('.app-container') as HTMLElement;
const activityBarButtons = document.querySelectorAll('.activity-bar button');
const sidebarPane = document.querySelector('.sidebar-pane') as HTMLElement;
const workspaceViewport = document.querySelector('.workspace-viewport') as HTMLElement;
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const resumeView = document.querySelector('.resume-view') as HTMLElement;

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

// --- VIEW SWITCHING LOGIC ---
function updateView(view) {
    // Update container attribute for CSS targeting
    appContainer.dataset.activeView = view;

    // Update active button state
    activityBarButtons.forEach(button => {
        button.classList.toggle('active', button.dataset.view === view);
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
        const view = button.dataset.view;
        if (view) {
            updateView(view);
        }
    });
});

// --- INITIALIZE ---
// Set default view
updateView('intro');

// Initialize WebGPU Cube
if (canvas) {
    const cube = new RotatingCube(canvas);

    // Resize canvas to fit its container
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.contentRect.width;
            const height = entry.contentRect.height;
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
                // Manually trigger a render to avoid stretching during resize, as
                // the main render loop might be paused by the browser.
                cube.render();
            }
        }
    });
    resizeObserver.observe(workspaceViewport);
    cube.start();
} else {
    console.error("Canvas element not found!");
}
