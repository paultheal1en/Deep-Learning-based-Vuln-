static uint32_t pxa2xx_gpio_read(void *opaque, target_phys_addr_t offset)
{
    struct pxa2xx_gpio_info_s *s = (struct pxa2xx_gpio_info_s *) opaque;
    uint32_t ret;
    int bank;
    offset -= s->base;
    if (offset >= 0x200)
        return 0;
    bank = pxa2xx_gpio_regs[offset].bank;
    switch (pxa2xx_gpio_regs[offset].reg) {
    case GPDR:		/* GPIO Pin-Direction registers */
        return s->dir[bank];
    case GRER:		/* GPIO Rising-Edge Detect Enable registers */
        return s->rising[bank];
    case GFER:		/* GPIO Falling-Edge Detect Enable registers */
        return s->falling[bank];
    case GAFR_L:	/* GPIO Alternate Function registers */
        return s->gafr[bank * 2];
    case GAFR_U:	/* GPIO Alternate Function registers */
        return s->gafr[bank * 2 + 1];
    case GPLR:		/* GPIO Pin-Level registers */
        ret = (s->olevel[bank] & s->dir[bank]) |
                (s->ilevel[bank] & ~s->dir[bank]);
        if (s->read_notify)
            s->read_notify(s->opaque);
        return ret;
    case GEDR:		/* GPIO Edge Detect Status registers */
        return s->status[bank];
    default:
        cpu_abort(cpu_single_env,
                "%s: Bad offset " REG_FMT "\n", __FUNCTION__, offset);
    }
    return 0;
}