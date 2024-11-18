static void pxa2xx_cm_write(void *opaque, hwaddr addr,

                            uint64_t value, unsigned size)

{

    PXA2xxState *s = (PXA2xxState *) opaque;



    switch (addr) {

    case CCCR:

    case CKEN:

        s->cm_regs[addr >> 2] = value;

        break;



    case OSCC:

        s->cm_regs[addr >> 2] &= ~0x6c;

        s->cm_regs[addr >> 2] |= value & 0x6e;

        if ((value >> 1) & 1)			/* OON */

            s->cm_regs[addr >> 2] |= 1 << 0;	/* Oscillator is now stable */

        break;



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

}
