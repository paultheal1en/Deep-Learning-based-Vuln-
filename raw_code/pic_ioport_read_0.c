static uint64_t pic_ioport_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    PICCommonState *s = opaque;

    int ret;



    if (s->poll) {

        ret = pic_get_irq(s);

        if (ret >= 0) {

            pic_intack(s, ret);

            ret |= 0x80;

        } else {

            ret = 0;

        }

        s->poll = 0;

    } else {

        if (addr == 0) {

            if (s->read_reg_select) {

                ret = s->isr;

            } else {

                ret = s->irr;

            }

        } else {

            ret = s->imr;

        }

    }

    DPRINTF("read: addr=0x%02x val=0x%02x\n", addr, ret);

    return ret;

}
