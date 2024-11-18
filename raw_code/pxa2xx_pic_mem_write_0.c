static void pxa2xx_pic_mem_write(void *opaque, hwaddr offset,

                                 uint64_t value, unsigned size)

{

    PXA2xxPICState *s = (PXA2xxPICState *) opaque;



    switch (offset) {

    case ICMR:	/* Mask register */

        s->int_enabled[0] = value;

        break;

    case ICMR2:	/* Mask register 2 */

        s->int_enabled[1] = value;

        break;

    case ICLR:	/* Level register */

        s->is_fiq[0] = value;

        break;

    case ICLR2:	/* Level register 2 */

        s->is_fiq[1] = value;

        break;

    case ICCR:	/* Idle mask */

        s->int_idle = (value & 1) ? 0 : ~0;

        break;

    case IPR0  ... IPR31:

        s->priority[0  + ((offset - IPR0 ) >> 2)] = value & 0x8000003f;

        break;

    case IPR32 ... IPR39:

        s->priority[32 + ((offset - IPR32) >> 2)] = value & 0x8000003f;

        break;

    default:

        printf("%s: Bad register offset " REG_FMT "\n", __FUNCTION__, offset);

        return;

    }

    pxa2xx_pic_update(opaque);

}
