static int tcg_target_const_match(tcg_target_long val, TCGType type,

                                  const TCGArgConstraint *arg_ct)

{

    int ct = arg_ct->ct;



    if (ct & TCG_CT_CONST) {

        return 1;

    }



    if (type == TCG_TYPE_I32) {

        val = (int32_t)val;

    }



    /* The following are mutually exclusive.  */

    if (ct & TCG_CT_CONST_S16) {

        return val == (int16_t)val;

    } else if (ct & TCG_CT_CONST_S32) {

        return val == (int32_t)val;

    } else if (ct & TCG_CT_CONST_ADLI) {

        return tcg_match_add2i(type, val);

    } else if (ct & TCG_CT_CONST_ORI) {

        return tcg_match_ori(type, val);

    } else if (ct & TCG_CT_CONST_XORI) {

        return tcg_match_xori(type, val);

    } else if (ct & TCG_CT_CONST_U31) {

        return val >= 0 && val <= 0x7fffffff;

    } else if (ct & TCG_CT_CONST_ZERO) {

        return val == 0;

    }



    return 0;

}
