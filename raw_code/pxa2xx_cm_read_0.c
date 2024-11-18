static uint64_t pxa2xx_cm_read(void *opaque, hwaddr addr,

                               unsigned size)

{

    PXA2xxState *s = (PXA2xxState *) opaque;



    switch (addr) {

    case CCCR:

    case CKEN:

    case OSCC:

        return s->cm_regs[addr >> 2];



    case CCSR:

        return s->cm_regs[CCCR >> 2] | (3 << 28);



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
