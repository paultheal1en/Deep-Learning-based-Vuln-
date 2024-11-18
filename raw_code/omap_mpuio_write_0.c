static void omap_mpuio_write(void *opaque, target_phys_addr_t addr,

                             uint64_t value, unsigned size)

{

    struct omap_mpuio_s *s = (struct omap_mpuio_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;

    uint16_t diff;

    int ln;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (offset) {

    case 0x04:	/* OUTPUT_REG */

        diff = (s->outputs ^ value) & ~s->dir;

        s->outputs = value;

        while ((ln = ffs(diff))) {

            ln --;

            if (s->handler[ln])

                qemu_set_irq(s->handler[ln], (value >> ln) & 1);

            diff &= ~(1 << ln);

        }

        break;



    case 0x08:	/* IO_CNTL */

        diff = s->outputs & (s->dir ^ value);

        s->dir = value;



        value = s->outputs & ~s->dir;

        while ((ln = ffs(diff))) {

            ln --;

            if (s->handler[ln])

                qemu_set_irq(s->handler[ln], (value >> ln) & 1);

            diff &= ~(1 << ln);

        }

        break;



    case 0x14:	/* KBC_REG */

        s->cols = value;

        omap_mpuio_kbd_update(s);

        break;



    case 0x18:	/* GPIO_EVENT_MODE_REG */

        s->event = value & 0x1f;

        break;



    case 0x1c:	/* GPIO_INT_EDGE_REG */

        s->edge = value;

        break;



    case 0x28:	/* KBD_MASKIT */

        s->kbd_mask = value & 1;

        omap_mpuio_kbd_update(s);

        break;



    case 0x2c:	/* GPIO_MASKIT */

        s->mask = value;

        break;



    case 0x30:	/* GPIO_DEBOUNCING_REG */

        s->debounce = value & 0x1ff;

        break;



    case 0x00:	/* INPUT_LATCH */

    case 0x10:	/* KBR_LATCH */

    case 0x20:	/* KBD_INT */

    case 0x24:	/* GPIO_INT */

    case 0x34:	/* GPIO_LATCH_REG */

        OMAP_RO_REG(addr);

        return;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
