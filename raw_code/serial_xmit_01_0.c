static void serial_xmit(void *opaque)

{

    SerialState *s = opaque;

    uint64_t new_xmit_ts = qemu_get_clock_ns(vm_clock);



    if (s->tsr_retry <= 0) {

        if (s->fcr & UART_FCR_FE) {

            s->tsr = fifo_get(s,XMIT_FIFO);

            if (!s->xmit_fifo.count)

                s->lsr |= UART_LSR_THRE;

        } else {

            s->tsr = s->thr;

            s->lsr |= UART_LSR_THRE;

        }

    }



    if (s->mcr & UART_MCR_LOOP) {

        /* in loopback mode, say that we just received a char */

        serial_receive1(s, &s->tsr, 1);

    } else if (qemu_chr_fe_write(s->chr, &s->tsr, 1) != 1) {

        if ((s->tsr_retry > 0) && (s->tsr_retry <= MAX_XMIT_RETRY)) {

            s->tsr_retry++;

            qemu_mod_timer(s->transmit_timer,  new_xmit_ts + s->char_transmit_time);

            return;

        } else if (s->poll_msl < 0) {

            /* If we exceed MAX_XMIT_RETRY and the backend is not a real serial port, then

            drop any further failed writes instantly, until we get one that goes through.

            This is to prevent guests that log to unconnected pipes or pty's from stalling. */

            s->tsr_retry = -1;

        }

    }

    else {

        s->tsr_retry = 0;

    }



    s->last_xmit_ts = qemu_get_clock_ns(vm_clock);

    if (!(s->lsr & UART_LSR_THRE))

        qemu_mod_timer(s->transmit_timer, s->last_xmit_ts + s->char_transmit_time);



    if (s->lsr & UART_LSR_THRE) {

        s->lsr |= UART_LSR_TEMT;

        s->thr_ipending = 1;

        serial_update_irq(s);

    }

}
