static void gen_nop_hint(DisasContext *s, int val)

{

    switch (val) {

    case 1: /* yield */

        if (!parallel_cpus) {

            gen_set_pc_im(s, s->pc);

            s->base.is_jmp = DISAS_YIELD;

        }

        break;

    case 3: /* wfi */

        gen_set_pc_im(s, s->pc);

        s->base.is_jmp = DISAS_WFI;

        break;

    case 2: /* wfe */

        if (!parallel_cpus) {

            gen_set_pc_im(s, s->pc);

            s->base.is_jmp = DISAS_WFE;

        }

        break;

    case 4: /* sev */

    case 5: /* sevl */

        /* TODO: Implement SEV, SEVL and WFE.  May help SMP performance.  */

    default: /* nop */

        break;

    }

}
