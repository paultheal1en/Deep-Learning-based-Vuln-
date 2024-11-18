static void serial_ioport_write(void *opaque, hwaddr addr, uint64_t val,

                                unsigned size)

{

    SerialState *s = opaque;



    addr &= 7;

    DPRINTF("write addr=0x%" HWADDR_PRIx " val=0x%" PRIx64 "\n", addr, val);

    switch(addr) {

    default:

    case 0:

        if (s->lcr & UART_LCR_DLAB) {

            s->divider = (s->divider & 0xff00) | val;

            serial_update_parameters(s);

        } else {

            s->thr = (uint8_t) val;

            if(s->fcr & UART_FCR_FE) {

                /* xmit overruns overwrite data, so make space if needed */

                if (fifo8_is_full(&s->xmit_fifo)) {

                    fifo8_pop(&s->xmit_fifo);

                }

                fifo8_push(&s->xmit_fifo, s->thr);

                s->lsr &= ~UART_LSR_TEMT;

            }

            s->thr_ipending = 0;

            s->lsr &= ~UART_LSR_THRE;

            serial_update_irq(s);

            serial_xmit(NULL, G_IO_OUT, s);

        }

        break;

    case 1:

        if (s->lcr & UART_LCR_DLAB) {

            s->divider = (s->divider & 0x00ff) | (val << 8);

            serial_update_parameters(s);

        } else {

            s->ier = val & 0x0f;

            /* If the backend device is a real serial port, turn polling of the modem

               status lines on physical port on or off depending on UART_IER_MSI state */

            if (s->poll_msl >= 0) {

                if (s->ier & UART_IER_MSI) {

                     s->poll_msl = 1;

                     serial_update_msl(s);

                } else {

                     timer_del(s->modem_status_poll);

                     s->poll_msl = 0;

                }

            }

            if (s->lsr & UART_LSR_THRE) {

                s->thr_ipending = 1;

                serial_update_irq(s);

            }

        }

        break;

    case 2:

        val = val & 0xFF;



        if (s->fcr == val)

            break;



        /* Did the enable/disable flag change? If so, make sure FIFOs get flushed */

        if ((val ^ s->fcr) & UART_FCR_FE)

            val |= UART_FCR_XFR | UART_FCR_RFR;



        /* FIFO clear */



        if (val & UART_FCR_RFR) {

            timer_del(s->fifo_timeout_timer);

            s->timeout_ipending=0;

            fifo8_reset(&s->recv_fifo);

        }



        if (val & UART_FCR_XFR) {

            fifo8_reset(&s->xmit_fifo);

        }



        if (val & UART_FCR_FE) {

            s->iir |= UART_IIR_FE;

            /* Set recv_fifo trigger Level */

            switch (val & 0xC0) {

            case UART_FCR_ITL_1:

                s->recv_fifo_itl = 1;

                break;

            case UART_FCR_ITL_2:

                s->recv_fifo_itl = 4;

                break;

            case UART_FCR_ITL_3:

                s->recv_fifo_itl = 8;

                break;

            case UART_FCR_ITL_4:

                s->recv_fifo_itl = 14;

                break;

            }

        } else

            s->iir &= ~UART_IIR_FE;



        /* Set fcr - or at least the bits in it that are supposed to "stick" */

        s->fcr = val & 0xC9;

        serial_update_irq(s);

        break;

    case 3:

        {

            int break_enable;

            s->lcr = val;

            serial_update_parameters(s);

            break_enable = (val >> 6) & 1;

            if (break_enable != s->last_break_enable) {

                s->last_break_enable = break_enable;

                qemu_chr_fe_ioctl(s->chr, CHR_IOCTL_SERIAL_SET_BREAK,

                               &break_enable);

            }

        }

        break;

    case 4:

        {

            int flags;

            int old_mcr = s->mcr;

            s->mcr = val & 0x1f;

            if (val & UART_MCR_LOOP)

                break;



            if (s->poll_msl >= 0 && old_mcr != s->mcr) {



                qemu_chr_fe_ioctl(s->chr,CHR_IOCTL_SERIAL_GET_TIOCM, &flags);



                flags &= ~(CHR_TIOCM_RTS | CHR_TIOCM_DTR);



                if (val & UART_MCR_RTS)

                    flags |= CHR_TIOCM_RTS;

                if (val & UART_MCR_DTR)

                    flags |= CHR_TIOCM_DTR;



                qemu_chr_fe_ioctl(s->chr,CHR_IOCTL_SERIAL_SET_TIOCM, &flags);

                /* Update the modem status after a one-character-send wait-time, since there may be a response

                   from the device/computer at the other end of the serial line */

                timer_mod(s->modem_status_poll, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + s->char_transmit_time);

            }

        }

        break;

    case 5:

        break;

    case 6:

        break;

    case 7:

        s->scr = val;

        break;

    }

}
