static uint64_t omap_ulpd_pm_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;

    uint16_t ret;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    switch (addr) {

    case 0x14:	/* IT_STATUS */

        ret = s->ulpd_pm_regs[addr >> 2];

        s->ulpd_pm_regs[addr >> 2] = 0;

        qemu_irq_lower(s->irq[1][OMAP_INT_GAUGE_32K]);

        return ret;



    case 0x18:	/* Reserved */

    case 0x1c:	/* Reserved */

    case 0x20:	/* Reserved */

    case 0x28:	/* Reserved */

    case 0x2c:	/* Reserved */

        OMAP_BAD_REG(addr);

    case 0x00:	/* COUNTER_32_LSB */

    case 0x04:	/* COUNTER_32_MSB */

    case 0x08:	/* COUNTER_HIGH_FREQ_LSB */

    case 0x0c:	/* COUNTER_HIGH_FREQ_MSB */

    case 0x10:	/* GAUGING_CTRL */

    case 0x24:	/* SETUP_ANALOG_CELL3_ULPD1 */

    case 0x30:	/* CLOCK_CTRL */

    case 0x34:	/* SOFT_REQ */

    case 0x38:	/* COUNTER_32_FIQ */

    case 0x3c:	/* DPLL_CTRL */

    case 0x40:	/* STATUS_REQ */

        /* XXX: check clk::usecount state for every clock */

    case 0x48:	/* LOCL_TIME */

    case 0x4c:	/* APLL_CTRL */

    case 0x50:	/* POWER_CTRL */

        return s->ulpd_pm_regs[addr >> 2];

    }



    OMAP_BAD_REG(addr);

    return 0;

}
