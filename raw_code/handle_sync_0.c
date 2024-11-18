static void handle_sync(DisasContext *s, uint32_t insn,

                        unsigned int op1, unsigned int op2, unsigned int crm)

{

    if (op1 != 3) {

        unallocated_encoding(s);

        return;

    }



    switch (op2) {

    case 2: /* CLREX */

        gen_clrex(s, insn);

        return;

    case 4: /* DSB */

    case 5: /* DMB */

    case 6: /* ISB */

        /* We don't emulate caches so barriers are no-ops */

        return;

    default:

        unallocated_encoding(s);

        return;

    }

}
