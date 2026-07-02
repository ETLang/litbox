export class ModalDialog {
    private overlay: HTMLElement;
    private modalContent: HTMLElement;
    private static instance: ModalDialog;

    private constructor() {
        this.overlay = document.getElementById('contact-modal-overlay') as HTMLElement;
        this.modalContent = this.overlay.querySelector('.modal-content') as HTMLElement;

        if (!this.overlay || !this.modalContent) {
            throw new Error('Modal elements not found in the DOM');
        }

        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hide();
            }
        });

        this.modalContent.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            if (target.id === 'ok-button') {
                this.hide();
            }
        });

        this.modalContent.addEventListener('submit', async (e) => {
            if (!(e.target instanceof HTMLFormElement) || !e.target.classList.contains('contact-form')) {
                return;
            }

            e.preventDefault();

            const form = e.target;
            const status = form.querySelector('.form-status') as HTMLElement;
            if (!status) return;

            const data = new FormData(form);
            status.innerHTML = 'Sending...';
            status.className = 'form-status sending';

            try {
                const response = await fetch(form.action, {
                    method: form.method,
                    body: data,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                if (response.ok) {
                    form.reset();
                    sessionStorage.removeItem('contactFormEmail');
                    sessionStorage.removeItem('contactFormMessage');
                    this.modalContent.innerHTML = `
                        <div class="form-success-message">
                            <p>Thanks for your submission!</p>
                            <button id="ok-button">OK</button>
                        </div>`;
                } else {
                    const responseData = await response.json();
                    status.innerHTML = responseData.errors?.map((error: { message: string }) => error.message).join(", ") || "Oops! There was a problem submitting your form.";
                    status.className = 'form-status error';
                }
            } catch (error) {
                console.error('Form submission error:', error);
                status.innerHTML = "Oops! There was a problem submitting your form.";
                status.className = 'form-status error';
            }
        });
    }

    public static getInstance(): ModalDialog {
        if (!ModalDialog.instance) {
            ModalDialog.instance = new ModalDialog();
        }
        return ModalDialog.instance;
    }

    public show(content: string) {
        // Only set content if it's not already there, to preserve form state
        if (this.modalContent.innerHTML !== content) {
            this.modalContent.innerHTML = content;
        }
        this.overlay.style.display = 'flex';

        // Restore form state
        const emailInput = this.modalContent.querySelector('#email') as HTMLInputElement;
        const messageInput = this.modalContent.querySelector('#message') as HTMLTextAreaElement;

        if (emailInput && messageInput) {
            emailInput.value = sessionStorage.getItem('contactFormEmail') || '';
            messageInput.value = sessionStorage.getItem('contactFormMessage') || '';
        }
    }

    public hide() {
        // Save form state
        const emailInput = this.modalContent.querySelector('#email') as HTMLInputElement;
        const messageInput = this.modalContent.querySelector('#message') as HTMLTextAreaElement;

        if (emailInput && messageInput) {
            sessionStorage.setItem('contactFormEmail', emailInput.value);
            sessionStorage.setItem('contactFormMessage', messageInput.value);
        }

        this.overlay.style.display = 'none';
    }
}
