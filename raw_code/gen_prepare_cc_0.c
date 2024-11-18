static CCPrepare gen_prepare_cc(DisasContext *s, int b, TCGv reg)

{

    int inv, jcc_op, size, cond;

    CCPrepare cc;

    TCGv t0;



    inv = b & 1;

    jcc_op = (b >> 1) & 7;



    switch (s->cc_op) {

    case CC_OP_SUBB ... CC_OP_SUBQ:

        /* We optimize relational operators for the cmp/jcc case.  */

        size = s->cc_op - CC_OP_SUBB;

        switch (jcc_op) {

        case JCC_BE:

            tcg_gen_add_tl(cpu_tmp4, cpu_cc_dst, cpu_cc_src);

            gen_extu(size, cpu_tmp4);

            t0 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, false);

            cc = (CCPrepare) { .cond = TCG_COND_LEU, .reg = cpu_tmp4,

                               .reg2 = t0, .mask = -1, .use_reg2 = true };

            break;



        case JCC_L:

            cond = TCG_COND_LT;

            goto fast_jcc_l;

        case JCC_LE:

            cond = TCG_COND_LE;

        fast_jcc_l:

            tcg_gen_add_tl(cpu_tmp4, cpu_cc_dst, cpu_cc_src);

            gen_exts(size, cpu_tmp4);

            t0 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, true);

            cc = (CCPrepare) { .cond = cond, .reg = cpu_tmp4,

                               .reg2 = t0, .mask = -1, .use_reg2 = true };

            break;



        default:

            goto slow_jcc;

        }

        break;



    default:

    slow_jcc:

        /* This actually generates good code for JC, JZ and JS.  */

        switch (jcc_op) {

        case JCC_O:

            cc = gen_prepare_eflags_o(s, reg);

            break;

        case JCC_B:

            cc = gen_prepare_eflags_c(s, reg);

            break;

        case JCC_Z:

            cc = gen_prepare_eflags_z(s, reg);

            break;

        case JCC_BE:

            gen_compute_eflags(s);

            cc = (CCPrepare) { .cond = TCG_COND_NE, .reg = cpu_cc_src,

                               .mask = CC_Z | CC_C };

            break;

        case JCC_S:

            cc = gen_prepare_eflags_s(s, reg);

            break;

        case JCC_P:

            cc = gen_prepare_eflags_p(s, reg);

            break;

        case JCC_L:

            gen_compute_eflags(s);

            if (TCGV_EQUAL(reg, cpu_cc_src)) {

                reg = cpu_tmp0;

            }

            tcg_gen_shri_tl(reg, cpu_cc_src, 4); /* CC_O -> CC_S */

            tcg_gen_xor_tl(reg, reg, cpu_cc_src);

            cc = (CCPrepare) { .cond = TCG_COND_NE, .reg = reg,

                               .mask = CC_S };

            break;

        default:

        case JCC_LE:

            gen_compute_eflags(s);

            if (TCGV_EQUAL(reg, cpu_cc_src)) {

                reg = cpu_tmp0;

            }

            tcg_gen_shri_tl(reg, cpu_cc_src, 4); /* CC_O -> CC_S */

            tcg_gen_xor_tl(reg, reg, cpu_cc_src);

            cc = (CCPrepare) { .cond = TCG_COND_NE, .reg = reg,

                               .mask = CC_S | CC_Z };

            break;

        }

        break;

    }



    if (inv) {

        cc.cond = tcg_invert_cond(cc.cond);

    }

    return cc;

}
