static void do_transmit_packets(dp8393xState *s)

{

    uint16_t data[12];

    int width, size;

    int tx_len, len;

    uint16_t i;



    width = (s->regs[SONIC_DCR] & SONIC_DCR_DW) ? 2 : 1;



    while (1) {

        /* Read memory */

        DPRINTF("Transmit packet at %08x\n",

                (s->regs[SONIC_UTDA] << 16) | s->regs[SONIC_CTDA]);

        size = sizeof(uint16_t) * 6 * width;

        s->regs[SONIC_TTDA] = s->regs[SONIC_CTDA];

        s->memory_rw(s->mem_opaque,

            ((s->regs[SONIC_UTDA] << 16) | s->regs[SONIC_TTDA]) + sizeof(uint16_t) * width,

            (uint8_t *)data, size, 0);

        tx_len = 0;



        /* Update registers */

        s->regs[SONIC_TCR] = data[0 * width] & 0xf000;

        s->regs[SONIC_TPS] = data[1 * width];

        s->regs[SONIC_TFC] = data[2 * width];

        s->regs[SONIC_TSA0] = data[3 * width];

        s->regs[SONIC_TSA1] = data[4 * width];

        s->regs[SONIC_TFS] = data[5 * width];



        /* Handle programmable interrupt */

        if (s->regs[SONIC_TCR] & SONIC_TCR_PINT) {

            s->regs[SONIC_ISR] |= SONIC_ISR_PINT;

        } else {

            s->regs[SONIC_ISR] &= ~SONIC_ISR_PINT;

        }



        for (i = 0; i < s->regs[SONIC_TFC]; ) {

            /* Append fragment */

            len = s->regs[SONIC_TFS];

            if (tx_len + len > sizeof(s->tx_buffer)) {

                len = sizeof(s->tx_buffer) - tx_len;

            }

            s->memory_rw(s->mem_opaque,

                (s->regs[SONIC_TSA1] << 16) | s->regs[SONIC_TSA0],

                &s->tx_buffer[tx_len], len, 0);

            tx_len += len;



            i++;

            if (i != s->regs[SONIC_TFC]) {

                /* Read next fragment details */

                size = sizeof(uint16_t) * 3 * width;

                s->memory_rw(s->mem_opaque,

                    ((s->regs[SONIC_UTDA] << 16) | s->regs[SONIC_TTDA]) + sizeof(uint16_t) * (4 + 3 * i) * width,

                    (uint8_t *)data, size, 0);

                s->regs[SONIC_TSA0] = data[0 * width];

                s->regs[SONIC_TSA1] = data[1 * width];

                s->regs[SONIC_TFS] = data[2 * width];

            }

        }



        /* Handle Ethernet checksum */

        if (!(s->regs[SONIC_TCR] & SONIC_TCR_CRCI)) {

            /* Don't append FCS there, to look like slirp packets

             * which don't have one */

        } else {

            /* Remove existing FCS */

            tx_len -= 4;

        }



        if (s->regs[SONIC_RCR] & (SONIC_RCR_LB1 | SONIC_RCR_LB0)) {

            /* Loopback */

            s->regs[SONIC_TCR] |= SONIC_TCR_CRSL;

            if (s->vc->fd_can_read(s)) {

                s->loopback_packet = 1;

                s->vc->receive(s, s->tx_buffer, tx_len);

            }

        } else {

            /* Transmit packet */

            qemu_send_packet(s->vc, s->tx_buffer, tx_len);

        }

        s->regs[SONIC_TCR] |= SONIC_TCR_PTX;



        /* Write status */

        data[0 * width] = s->regs[SONIC_TCR] & 0x0fff; /* status */

        size = sizeof(uint16_t) * width;

        s->memory_rw(s->mem_opaque,

            (s->regs[SONIC_UTDA] << 16) | s->regs[SONIC_TTDA],

            (uint8_t *)data, size, 1);



        if (!(s->regs[SONIC_CR] & SONIC_CR_HTX)) {

            /* Read footer of packet */

            size = sizeof(uint16_t) * width;

            s->memory_rw(s->mem_opaque,

                ((s->regs[SONIC_UTDA] << 16) | s->regs[SONIC_TTDA]) + sizeof(uint16_t) * (4 + 3 * s->regs[SONIC_TFC]) * width,

                (uint8_t *)data, size, 0);

            s->regs[SONIC_CTDA] = data[0 * width] & ~0x1;

            if (data[0 * width] & 0x1) {

                /* EOL detected */

                break;

            }

        }

    }



    /* Done */

    s->regs[SONIC_CR] &= ~SONIC_CR_TXP;

    s->regs[SONIC_ISR] |= SONIC_ISR_TXDN;

    dp8393x_update_irq(s);

}
