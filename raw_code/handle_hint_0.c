static void handle_hint(DisasContext *s, uint32_t insn,

                        unsigned int op1, unsigned int op2, unsigned int crm)

{

    unsigned int selector = crm << 3 | op2;



    if (op1 != 3) {

        unallocated_encoding(s);

        return;

    }



    switch (selector) {

    case 0: /* NOP */

        return;

    case 3: /* WFI */

        s->base.is_jmp = DISAS_WFI;

        return;

    case 1: /* YIELD */

        if (!parallel_cpus) {

            s->base.is_jmp = DISAS_YIELD;

        }

        return;

    case 2: /* WFE */

        if (!parallel_cpus) {

            s->base.is_jmp = DISAS_WFE;

        }

        return;

    case 4: /* SEV */

    case 5: /* SEVL */

        /* we treat all as NOP at least for now */

        return;

    default:

        /* default specified as NOP equivalent */

        return;

    }

}
