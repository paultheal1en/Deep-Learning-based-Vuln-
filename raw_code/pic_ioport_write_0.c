static void pic_ioport_write(void *opaque, target_phys_addr_t addr64,

                             uint64_t val64, unsigned size)

{

    PICCommonState *s = opaque;

    uint32_t addr = addr64;

    uint32_t val = val64;

    int priority, cmd, irq;



    DPRINTF("write: addr=0x%02x val=0x%02x\n", addr, val);

    if (addr == 0) {

        if (val & 0x10) {

            pic_init_reset(s);

            s->init_state = 1;

            s->init4 = val & 1;

            s->single_mode = val & 2;

            if (val & 0x08) {

                hw_error("level sensitive irq not supported");

            }

        } else if (val & 0x08) {

            if (val & 0x04) {

                s->poll = 1;

            }

            if (val & 0x02) {

                s->read_reg_select = val & 1;

            }

            if (val & 0x40) {

                s->special_mask = (val >> 5) & 1;

            }

        } else {

            cmd = val >> 5;

            switch (cmd) {

            case 0:

            case 4:

                s->rotate_on_auto_eoi = cmd >> 2;

                break;

            case 1: /* end of interrupt */

            case 5:

                priority = get_priority(s, s->isr);

                if (priority != 8) {

                    irq = (priority + s->priority_add) & 7;

                    s->isr &= ~(1 << irq);

                    if (cmd == 5) {

                        s->priority_add = (irq + 1) & 7;

                    }

                    pic_update_irq(s);

                }

                break;

            case 3:

                irq = val & 7;

                s->isr &= ~(1 << irq);

                pic_update_irq(s);

                break;

            case 6:

                s->priority_add = (val + 1) & 7;

                pic_update_irq(s);

                break;

            case 7:

                irq = val & 7;

                s->isr &= ~(1 << irq);

                s->priority_add = (irq + 1) & 7;

                pic_update_irq(s);

                break;

            default:

                /* no operation */

                break;

            }

        }

    } else {

        switch (s->init_state) {

        case 0:

            /* normal mode */

            s->imr = val;

            pic_update_irq(s);

            break;

        case 1:

            s->irq_base = val & 0xf8;

            s->init_state = s->single_mode ? (s->init4 ? 3 : 0) : 2;

            break;

        case 2:

            if (s->init4) {

                s->init_state = 3;

            } else {

                s->init_state = 0;

            }

            break;

        case 3:

            s->special_fully_nested_mode = (val >> 4) & 1;

            s->auto_eoi = (val >> 1) & 1;

            s->init_state = 0;

            break;

        }

    }

}
