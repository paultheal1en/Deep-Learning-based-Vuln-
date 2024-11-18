static inline int tcg_target_const_match(tcg_target_long val,

                                         const TCGArgConstraint *arg_ct)

{

    int ct;



    ct = arg_ct->ct;

    if (ct & TCG_CT_CONST)

        return 1;

    else if ((ct & TCG_CT_CONST_S11) && ABS(val) == (ABS(val) & 0x3ff))

        return 1;

    else if ((ct & TCG_CT_CONST_S13) && ABS(val) == (ABS(val) & 0xfff))

        return 1;

    else

        return 0;

}
