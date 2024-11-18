static uint64_t pxa2xx_fir_read(void *opaque, hwaddr addr,

                                unsigned size)

{

    PXA2xxFIrState *s = (PXA2xxFIrState *) opaque;

    uint8_t ret;



    switch (addr) {

    case ICCR0:

        return s->control[0];

    case ICCR1:

        return s->control[1];

    case ICCR2:

        return s->control[2];

    case ICDR:

        s->status[0] &= ~0x01;

        s->status[1] &= ~0x72;

        if (s->rx_len) {

            s->rx_len --;

            ret = s->rx_fifo[s->rx_start ++];

            s->rx_start &= 63;

            pxa2xx_fir_update(s);

            return ret;

        }

        printf("%s: Rx FIFO underrun.\n", __FUNCTION__);

        break;

    case ICSR0:

        return s->status[0];

    case ICSR1:

        return s->status[1] | (1 << 3);			/* TNF */

    case ICFOR:

        return s->rx_len;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
