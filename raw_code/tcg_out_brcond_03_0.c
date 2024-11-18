static void tcg_out_brcond(TCGContext *s, TCGCond cond,

                           TCGArg arg1, TCGArg arg2, int const_arg2,

                           int label_index, TCGType type)

{

    tcg_out_cmp(s, cond, arg1, arg2, const_arg2, 7, type);

    tcg_out_bc(s, tcg_to_bc[cond], label_index);

}
