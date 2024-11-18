static void serial_update_irq(SerialState *s)

{

    uint8_t tmp_iir = UART_IIR_NO_INT;



    if ((s->ier & UART_IER_RLSI) && (s->lsr & UART_LSR_INT_ANY)) {

        tmp_iir = UART_IIR_RLSI;

    } else if ((s->ier & UART_IER_RDI) && s->timeout_ipending) {

        /* Note that(s->ier & UART_IER_RDI) can mask this interrupt,

         * this is not in the specification but is observed on existing

         * hardware.  */

        tmp_iir = UART_IIR_CTI;

    } else if ((s->ier & UART_IER_RDI) && (s->lsr & UART_LSR_DR)) {

        if (!(s->fcr & UART_FCR_FE)) {

           tmp_iir = UART_IIR_RDI;

        } else if (s->recv_fifo.count >= s->recv_fifo.itl) {

           tmp_iir = UART_IIR_RDI;

        }

    } else if ((s->ier & UART_IER_THRI) && s->thr_ipending) {

        tmp_iir = UART_IIR_THRI;

    } else if ((s->ier & UART_IER_MSI) && (s->msr & UART_MSR_ANY_DELTA)) {

        tmp_iir = UART_IIR_MSI;

    }



    s->iir = tmp_iir | (s->iir & 0xF0);



    if (tmp_iir != UART_IIR_NO_INT) {

        qemu_irq_raise(s->irq);

    } else {

        qemu_irq_lower(s->irq);

    }

}
