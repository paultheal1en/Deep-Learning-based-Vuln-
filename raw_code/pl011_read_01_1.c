static uint64_t pl011_read(void *opaque, target_phys_addr_t offset,

                           unsigned size)

{

    pl011_state *s = (pl011_state *)opaque;

    uint32_t c;



    if (offset >= 0xfe0 && offset < 0x1000) {

        return s->id[(offset - 0xfe0) >> 2];

    }

    switch (offset >> 2) {

    case 0: /* UARTDR */

        s->flags &= ~PL011_FLAG_RXFF;

        c = s->read_fifo[s->read_pos];

        if (s->read_count > 0) {

            s->read_count--;

            if (++s->read_pos == 16)

                s->read_pos = 0;

        }

        if (s->read_count == 0) {

            s->flags |= PL011_FLAG_RXFE;

        }

        if (s->read_count == s->read_trigger - 1)

            s->int_level &= ~ PL011_INT_RX;

        pl011_update(s);

        qemu_chr_accept_input(s->chr);

        return c;

    case 1: /* UARTCR */

        return 0;

    case 6: /* UARTFR */

        return s->flags;

    case 8: /* UARTILPR */

        return s->ilpr;

    case 9: /* UARTIBRD */

        return s->ibrd;

    case 10: /* UARTFBRD */

        return s->fbrd;

    case 11: /* UARTLCR_H */

        return s->lcr;

    case 12: /* UARTCR */

        return s->cr;

    case 13: /* UARTIFLS */

        return s->ifl;

    case 14: /* UARTIMSC */

        return s->int_enabled;

    case 15: /* UARTRIS */

        return s->int_level;

    case 16: /* UARTMIS */

        return s->int_level & s->int_enabled;

    case 18: /* UARTDMACR */

        return s->dmacr;

    default:

        hw_error("pl011_read: Bad offset %x\n", (int)offset);

        return 0;

    }

}
