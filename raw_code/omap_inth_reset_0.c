void omap_inth_reset(struct omap_intr_handler_s *s)

{

    int i;



    for (i = 0; i < s->nbanks; ++i){

        s->bank[i].irqs = 0x00000000;

        s->bank[i].mask = 0xffffffff;

        s->bank[i].sens_edge = 0x00000000;

        s->bank[i].fiq = 0x00000000;

        s->bank[i].inputs = 0x00000000;

        s->bank[i].swi = 0x00000000;

        memset(s->bank[i].priority, 0, sizeof(s->bank[i].priority));



        if (s->level_only)

            s->bank[i].sens_edge = 0xffffffff;

    }



    s->new_agr[0] = ~0;

    s->new_agr[1] = ~0;

    s->sir_intr[0] = 0;

    s->sir_intr[1] = 0;

    s->autoidle = 0;

    s->mask = ~0;



    qemu_set_irq(s->parent_intr[0], 0);

    qemu_set_irq(s->parent_intr[1], 0);

}
