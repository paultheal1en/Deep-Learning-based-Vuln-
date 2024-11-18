static void tcg_out_brcond2 (TCGContext *s, const TCGArg *args,

                             const int *const_args)

{

    tcg_out_cmp2(s, args, const_args);

    tcg_out_bc(s, BC | BI(7, CR_EQ) | BO_COND_TRUE, args[5]);

}
