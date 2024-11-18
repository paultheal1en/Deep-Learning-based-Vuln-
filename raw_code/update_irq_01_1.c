static void update_irq(struct xlx_pic *p)

{

    uint32_t i;



    /* level triggered interrupt */

    if (p->regs[R_MER] & 2) {

        p->regs[R_ISR] |= p->irq_pin_state & ~p->c_kind_of_intr;

    }



    /* Update the pending register.  */

    p->regs[R_IPR] = p->regs[R_ISR] & p->regs[R_IER];



    /* Update the vector register.  */

    for (i = 0; i < 32; i++) {

        if (p->regs[R_IPR] & (1 << i))

            break;

    }

    if (i == 32)

        i = ~0;



    p->regs[R_IVR] = i;

    qemu_set_irq(p->parent_irq, (p->regs[R_MER] & 1) && p->regs[R_IPR]);

}
