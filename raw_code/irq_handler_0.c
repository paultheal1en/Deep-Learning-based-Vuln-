static void irq_handler(void *opaque, int irq, int level)

{

    struct xlx_pic *p = opaque;



    if (!(p->regs[R_MER] & 2)) {

        qemu_irq_lower(p->parent_irq);

        return;

    }



    /* edge triggered interrupt */

    if (p->c_kind_of_intr & (1 << irq) && p->regs[R_MER] & 2) {

        p->regs[R_ISR] |= (level << irq);

    }



    p->irq_pin_state &= ~(1 << irq);

    p->irq_pin_state |= level << irq;

    update_irq(p);

}
