static void omap_sti_write(void *opaque, target_phys_addr_t addr,

                           uint64_t value, unsigned size)

{

    struct omap_sti_s *s = (struct omap_sti_s *) opaque;



    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* STI_REVISION */

    case 0x14:	/* STI_SYSSTATUS / STI_RX_STATUS / XTI_SYSSTATUS */

        OMAP_RO_REG(addr);

        return;



    case 0x10:	/* STI_SYSCONFIG */

        if (value & (1 << 1))				/* SOFTRESET */

            omap_sti_reset(s);

        s->sysconfig = value & 0xfe;

        break;



    case 0x18:	/* STI_IRQSTATUS */

        s->irqst &= ~value;

        omap_sti_interrupt_update(s);

        break;



    case 0x1c:	/* STI_IRQSETEN / STI_IRQCLREN */

        s->irqen = value & 0xffff;

        omap_sti_interrupt_update(s);

        break;



    case 0x2c:	/* STI_CLK_CTRL / XTI_SCLKCRTL */

        s->clkcontrol = value & 0xff;

        break;



    case 0x30:	/* STI_SERIAL_CFG / XTI_SCONFIG */

        s->serial_config = value & 0xff;

        break;



    case 0x24:	/* STI_ER / STI_DR / XTI_TRACESELECT */

    case 0x28:	/* STI_RX_DR / XTI_RXDATA */

        /* TODO */

        return;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
