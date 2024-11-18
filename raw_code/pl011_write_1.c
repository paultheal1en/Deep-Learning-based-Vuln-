static void pl011_write(void *opaque, hwaddr offset,

                        uint64_t value, unsigned size)

{

    PL011State *s = (PL011State *)opaque;

    unsigned char ch;



    switch (offset >> 2) {

    case 0: /* UARTDR */

        /* ??? Check if transmitter is enabled.  */

        ch = value;

        if (s->chr)

            qemu_chr_fe_write(s->chr, &ch, 1);

        s->int_level |= PL011_INT_TX;

        pl011_update(s);

        break;

    case 1: /* UARTCR */

        s->cr = value;

        break;

    case 6: /* UARTFR */

        /* Writes to Flag register are ignored.  */

        break;

    case 8: /* UARTUARTILPR */

        s->ilpr = value;

        break;

    case 9: /* UARTIBRD */

        s->ibrd = value;

        break;

    case 10: /* UARTFBRD */

        s->fbrd = value;

        break;

    case 11: /* UARTLCR_H */

        /* Reset the FIFO state on FIFO enable or disable */

        if ((s->lcr ^ value) & 0x10) {

            s->read_count = 0;

            s->read_pos = 0;

        }

        s->lcr = value;

        pl011_set_read_trigger(s);

        break;

    case 12: /* UARTCR */

        /* ??? Need to implement the enable and loopback bits.  */

        s->cr = value;

        break;

    case 13: /* UARTIFS */

        s->ifl = value;

        pl011_set_read_trigger(s);

        break;

    case 14: /* UARTIMSC */

        s->int_enabled = value;

        pl011_update(s);

        break;

    case 17: /* UARTICR */

        s->int_level &= ~value;

        pl011_update(s);

        break;

    case 18: /* UARTDMACR */

        s->dmacr = value;

        if (value & 3) {

            qemu_log_mask(LOG_UNIMP, "pl011: DMA not implemented\n");

        }

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "pl011_write: Bad offset %x\n", (int)offset);

    }

}
