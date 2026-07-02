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
