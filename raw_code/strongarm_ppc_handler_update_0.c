static void strongarm_ppc_handler_update(StrongARMPPCInfo *s)

{

    uint32_t level, diff;

    int bit;



    level = s->olevel & s->dir;



    for (diff = s->prev_level ^ level; diff; diff ^= 1 << bit) {

        bit = ffs(diff) - 1;

        qemu_set_irq(s->handler[bit], (level >> bit) & 1);

    }



    s->prev_level = level;

}
