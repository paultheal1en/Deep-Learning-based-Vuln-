static uint32_t pxa2xx_pm_read(void *opaque, target_phys_addr_t addr)

{

    struct pxa2xx_state_s *s = (struct pxa2xx_state_s *) opaque;

    if (addr > s->pm_base + PCMD31) {

        /* Special case: PWRI2C registers appear in the same range.  */

        return pxa2xx_i2c_read(s->i2c[1], addr);

    }

    addr -= s->pm_base;



    switch (addr) {

    case PMCR ... PCMD31:

        if (addr & 3)

            goto fail;



        return s->pm_regs[addr >> 2];

    default:

    fail:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
