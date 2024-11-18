static void pl061_update(pl061_state *s)

{

    uint8_t changed;

    uint8_t mask;

    uint8_t out;

    int i;



    /* Outputs float high.  */

    /* FIXME: This is board dependent.  */

    out = (s->data & s->dir) | ~s->dir;

    changed = s->old_data ^ out;

    if (!changed)

        return;



    s->old_data = out;

    for (i = 0; i < 8; i++) {

        mask = 1 << i;

        if ((changed & mask) && s->out) {

            DPRINTF("Set output %d = %d\n", i, (out & mask) != 0);

            qemu_set_irq(s->out[i], (out & mask) != 0);

        }

    }



    /* FIXME: Implement input interrupts.  */

}
