static void icp_pic_write(void *opaque, target_phys_addr_t offset,

                          uint64_t value, unsigned size)

{

    icp_pic_state *s = (icp_pic_state *)opaque;



    switch (offset >> 2) {

    case 2: /* IRQ_ENABLESET */

        s->irq_enabled |= value;

        break;

    case 3: /* IRQ_ENABLECLR */

        s->irq_enabled &= ~value;

        break;

    case 4: /* INT_SOFTSET */

        if (value & 1)

            icp_pic_set_irq(s, 0, 1);

        break;

    case 5: /* INT_SOFTCLR */

        if (value & 1)

            icp_pic_set_irq(s, 0, 0);

        break;

    case 10: /* FRQ_ENABLESET */

        s->fiq_enabled |= value;

        break;

    case 11: /* FRQ_ENABLECLR */

        s->fiq_enabled &= ~value;

        break;

    case 0: /* IRQ_STATUS */

    case 1: /* IRQ_RAWSTAT */

    case 8: /* FRQ_STATUS */

    case 9: /* FRQ_RAWSTAT */

    default:

        printf ("icp_pic_write: Bad register offset 0x%x\n", (int)offset);

        return;

    }

    icp_pic_update(s);

}
