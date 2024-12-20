static void ne2000_ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    NE2000State *s = opaque;

    int offset, page;



    addr &= 0xf;

#ifdef DEBUG_NE2000

    printf("NE2000: write addr=0x%x val=0x%02x\n", addr, val);

#endif

    if (addr == E8390_CMD) {

        /* control register */

        s->cmd = val;

        if (val & E8390_START) {

            s->isr &= ~ENISR_RESET;

            /* test specific case: zero length transfert */

            if ((val & (E8390_RREAD | E8390_RWRITE)) &&

                s->rcnt == 0) {

                s->isr |= ENISR_RDC;

                ne2000_update_irq(s);

            }

            if (val & E8390_TRANS) {

                qemu_send_packet(s->nd, s->mem + (s->tpsr << 8), s->tcnt);

                /* signal end of transfert */

                s->tsr = ENTSR_PTX;

                s->isr |= ENISR_TX;

                ne2000_update_irq(s);

            }

        }

    } else {

        page = s->cmd >> 6;

        offset = addr | (page << 4);

        switch(offset) {

        case EN0_STARTPG:

            s->start = val << 8;

            break;

        case EN0_STOPPG:

            s->stop = val << 8;

            break;

        case EN0_BOUNDARY:

            s->boundary = val;

            break;

        case EN0_IMR:

            s->imr = val;

            ne2000_update_irq(s);

            break;

        case EN0_TPSR:

            s->tpsr = val;

            break;

        case EN0_TCNTLO:

            s->tcnt = (s->tcnt & 0xff00) | val;

            break;

        case EN0_TCNTHI:

            s->tcnt = (s->tcnt & 0x00ff) | (val << 8);

            break;

        case EN0_RSARLO:

            s->rsar = (s->rsar & 0xff00) | val;

            break;

        case EN0_RSARHI:

            s->rsar = (s->rsar & 0x00ff) | (val << 8);

            break;

        case EN0_RCNTLO:

            s->rcnt = (s->rcnt & 0xff00) | val;

            break;

        case EN0_RCNTHI:

            s->rcnt = (s->rcnt & 0x00ff) | (val << 8);

            break;

        case EN0_DCFG:

            s->dcfg = val;

            break;

        case EN0_ISR:

            s->isr &= ~(val & 0x7f);

            ne2000_update_irq(s);

            break;

        case EN1_PHYS ... EN1_PHYS + 5:

            s->phys[offset - EN1_PHYS] = val;

            break;

        case EN1_CURPAG:

            s->curpag = val;

            break;

        case EN1_MULT ... EN1_MULT + 7:

            s->mult[offset - EN1_MULT] = val;

            break;

        }

    }

}
