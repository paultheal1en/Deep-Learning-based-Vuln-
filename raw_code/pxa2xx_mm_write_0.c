static void pxa2xx_mm_write(void *opaque, hwaddr addr,

                            uint64_t value, unsigned size)

{

    PXA2xxState *s = (PXA2xxState *) opaque;



    switch (addr) {

    case MDCNFG ... SA1110:

        if ((addr & 3) == 0) {

            s->mm_regs[addr >> 2] = value;

            break;

        }



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

}
