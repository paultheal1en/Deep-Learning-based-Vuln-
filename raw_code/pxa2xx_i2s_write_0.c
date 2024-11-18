static void pxa2xx_i2s_write(void *opaque, hwaddr addr,

                             uint64_t value, unsigned size)

{

    PXA2xxI2SState *s = (PXA2xxI2SState *) opaque;

    uint32_t *sample;



    switch (addr) {

    case SACR0:

        if (value & (1 << 3))				/* RST */

            pxa2xx_i2s_reset(s);

        s->control[0] = value & 0xff3d;

        if (!s->enable && (value & 1) && s->tx_len) {	/* ENB */

            for (sample = s->fifo; s->fifo_len > 0; s->fifo_len --, sample ++)

                s->codec_out(s->opaque, *sample);

            s->status &= ~(1 << 7);			/* I2SOFF */

        }

        if (value & (1 << 4))				/* EFWR */

            printf("%s: Attempt to use special function\n", __FUNCTION__);

        s->enable = (value & 9) == 1;			/* ENB && !RST*/

        pxa2xx_i2s_update(s);

        break;

    case SACR1:

        s->control[1] = value & 0x0039;

        if (value & (1 << 5))				/* ENLBF */

            printf("%s: Attempt to use loopback function\n", __FUNCTION__);

        if (value & (1 << 4))				/* DPRL */

            s->fifo_len = 0;

        pxa2xx_i2s_update(s);

        break;

    case SAIMR:

        s->mask = value & 0x0078;

        pxa2xx_i2s_update(s);

        break;

    case SAICR:

        s->status &= ~(value & (3 << 5));

        pxa2xx_i2s_update(s);

        break;

    case SADIV:

        s->clk = value & 0x007f;

        break;

    case SADR:

        if (s->tx_len && s->enable) {

            s->tx_len --;

            pxa2xx_i2s_update(s);

            s->codec_out(s->opaque, value);

        } else if (s->fifo_len < 16) {

            s->fifo[s->fifo_len ++] = value;

            pxa2xx_i2s_update(s);

        }

        break;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

    }

}
