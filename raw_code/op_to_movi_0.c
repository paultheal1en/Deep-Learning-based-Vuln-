static int op_to_movi(int op)

{

    switch (op_bits(op)) {

    case 32:

        return INDEX_op_movi_i32;

#if TCG_TARGET_REG_BITS == 64

    case 64:

        return INDEX_op_movi_i64;

#endif

    default:

        fprintf(stderr, "op_to_movi: unexpected return value of "

                "function op_bits.\n");

        tcg_abort();

    }

}
