document.addEventListener('DOMContentLoaded', function() {
    // Using paths relative to the site root is more robust for deployment.
    fetch('/litbox/src/resumes/resume-specialist.md')
        .then(response => response.text())
        .then(text => {
            const resumeContainer = document.querySelector('.resume-view-specialist');
            if (resumeContainer) {
                resumeContainer.innerHTML = marked.parse(text);
            }
        });

    fetch('/litbox/src/resumes/resume-generalist.md')
        .then(response => response.text())
        .then(text => {
            const resumeContainer = document.querySelector('.resume-view-generalist');
            if (resumeContainer) {
                resumeContainer.innerHTML = marked.parse(text);
            }
        });
});
