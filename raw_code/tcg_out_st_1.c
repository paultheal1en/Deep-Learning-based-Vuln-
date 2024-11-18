static void tcg_out_st (TCGContext *s, TCGType type, int arg, int arg1,

                        tcg_target_long arg2)

{

    if (type == TCG_TYPE_I32)

        tcg_out_ldst (s, arg, arg1, arg2, STW, STWX);

    else

        tcg_out_ldst (s, arg, arg1, arg2, STD, STDX);

}
