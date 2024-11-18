static void tcg_out_ri32(TCGContext *s, int const_arg, TCGArg arg)

{

    if (const_arg) {

        assert(const_arg == 1);

        tcg_out8(s, TCG_CONST);

        tcg_out32(s, arg);

    } else {

        tcg_out_r(s, arg);

    }

}
