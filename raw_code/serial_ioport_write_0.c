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

            }

            s->thr_ipending = 0;

            s->lsr &= ~UART_LSR_THRE;

            s->lsr &= ~UART_LSR_TEMT;

            serial_update_irq(s);

            if (s->tsr_retry <= 0) {

                serial_xmit(NULL, G_IO_OUT, s);

            }

        }

        break;

    case 1:

        if (s->lcr & UART_LCR_DLAB) {

            s->divider = (s->divider & 0x00ff) | (val << 8);

            serial_update_parameters(s);

        } else {

            uint8_t changed = (s->ier ^ val) & 0x0f;

            s->ier = val & 0x0f;

            /* If the backend device is a real serial port, turn polling of the modem

             * status lines on physical port on or off depending on UART_IER_MSI state.

             */

            if ((changed & UART_IER_MSI) && s->poll_msl >= 0) {

                if (s->ier & UART_IER_MSI) {

                     s->poll_msl = 1;

                     serial_update_msl(s);

                } else {

                     timer_del(s->modem_status_poll);

                     s->poll_msl = 0;

                }

            }



            /* Turning on the THRE interrupt on IER can trigger the interrupt

             * if LSR.THRE=1, even if it had been masked before by reading IIR.

             * This is not in the datasheet, but Windows relies on it.  It is

             * unclear if THRE has to be resampled every time THRI becomes

             * 1, or only on the rising edge.  Bochs does the latter, and Windows

             * always toggles IER to all zeroes and back to all ones, so do the

             * same.

             *

             * If IER.THRI is zero, thr_ipending is not used.  Set it to zero

             * so that the thr_ipending subsection is not migrated.

             */

            if (changed & UART_IER_THRI) {

                if ((s->ier & UART_IER_THRI) && (s->lsr & UART_LSR_THRE)) {

                    s->thr_ipending = 1;

                } else {

                    s->thr_ipending = 0;

                }

            }



            if (changed) {

                serial_update_irq(s);

            }

        }

        break;

    case 2:

        /* Did the enable/disable flag change? If so, make sure FIFOs get flushed */

        if ((val ^ s->fcr) & UART_FCR_FE) {

            val |= UART_FCR_XFR | UART_FCR_RFR;

        }



        /* FIFO clear */



        if (val & UART_FCR_RFR) {

            s->lsr &= ~(UART_LSR_DR | UART_LSR_BI);

            timer_del(s->fifo_timeout_timer);

            s->timeout_ipending = 0;

            fifo8_reset(&s->recv_fifo);

        }



        if (val & UART_FCR_XFR) {

            s->lsr |= UART_LSR_THRE;

            s->thr_ipending = 1;

            fifo8_reset(&s->xmit_fifo);

        }



        serial_write_fcr(s, val & 0xC9);

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
