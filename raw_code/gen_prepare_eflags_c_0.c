static CCPrepare gen_prepare_eflags_c(DisasContext *s, TCGv reg)

{

    TCGv t0, t1;

    int size, shift;



    switch (s->cc_op) {

    case CC_OP_SUBB ... CC_OP_SUBQ:

        /* (DATA_TYPE)(CC_DST + CC_SRC) < (DATA_TYPE)CC_SRC */

        size = s->cc_op - CC_OP_SUBB;

        t1 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, false);

        /* If no temporary was used, be careful not to alias t1 and t0.  */

        t0 = TCGV_EQUAL(t1, cpu_cc_src) ? cpu_tmp0 : reg;

        tcg_gen_add_tl(t0, cpu_cc_dst, cpu_cc_src);

        gen_extu(size, t0);

        goto add_sub;



    case CC_OP_ADDB ... CC_OP_ADDQ:

        /* (DATA_TYPE)CC_DST < (DATA_TYPE)CC_SRC */

        size = s->cc_op - CC_OP_ADDB;

        t1 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, false);

        t0 = gen_ext_tl(reg, cpu_cc_dst, size, false);

    add_sub:

        return (CCPrepare) { .cond = TCG_COND_LTU, .reg = t0,

                             .reg2 = t1, .mask = -1, .use_reg2 = true };



    case CC_OP_SBBB ... CC_OP_SBBQ:

        /* (DATA_TYPE)(CC_DST + CC_SRC + 1) <= (DATA_TYPE)CC_SRC */

        size = s->cc_op - CC_OP_SBBB;

        t1 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, false);

        if (TCGV_EQUAL(t1, reg) && TCGV_EQUAL(reg, cpu_cc_src)) {

            tcg_gen_mov_tl(cpu_tmp0, cpu_cc_src);

            t1 = cpu_tmp0;

        }



        tcg_gen_add_tl(reg, cpu_cc_dst, cpu_cc_src);

        tcg_gen_addi_tl(reg, reg, 1);

        gen_extu(size, reg);

        t0 = reg;

        goto adc_sbb;



    case CC_OP_ADCB ... CC_OP_ADCQ:

        /* (DATA_TYPE)CC_DST <= (DATA_TYPE)CC_SRC */

        size = s->cc_op - CC_OP_ADCB;

        t1 = gen_ext_tl(cpu_tmp0, cpu_cc_src, size, false);

        t0 = gen_ext_tl(reg, cpu_cc_dst, size, false);

    adc_sbb:

        return (CCPrepare) { .cond = TCG_COND_LEU, .reg = t0,

                             .reg2 = t1, .mask = -1, .use_reg2 = true };



    case CC_OP_LOGICB ... CC_OP_LOGICQ:

        return (CCPrepare) { .cond = TCG_COND_NEVER, .mask = -1 };



    case CC_OP_INCB ... CC_OP_INCQ:

    case CC_OP_DECB ... CC_OP_DECQ:

        return (CCPrepare) { .cond = TCG_COND_NE, .reg = cpu_cc_src,

                             .mask = -1, .no_setcond = true };



    case CC_OP_SHLB ... CC_OP_SHLQ:

        /* (CC_SRC >> (DATA_BITS - 1)) & 1 */

        size = s->cc_op - CC_OP_SHLB;

        shift = (8 << size) - 1;

        return (CCPrepare) { .cond = TCG_COND_NE, .reg = cpu_cc_src,

                             .mask = (target_ulong)1 << shift };



    case CC_OP_MULB ... CC_OP_MULQ:

        return (CCPrepare) { .cond = TCG_COND_NE,

                             .reg = cpu_cc_src, .mask = -1 };



    case CC_OP_EFLAGS:

    case CC_OP_SARB ... CC_OP_SARQ:

        /* CC_SRC & 1 */

        return (CCPrepare) { .cond = TCG_COND_NE,

                             .reg = cpu_cc_src, .mask = CC_C };



    default:

       /* The need to compute only C from CC_OP_DYNAMIC is important

          in efficiently implementing e.g. INC at the start of a TB.  */

       gen_update_cc_op(s);

       gen_helper_cc_compute_c(cpu_tmp2_i32, cpu_env, cpu_cc_op);

       tcg_gen_extu_i32_tl(reg, cpu_tmp2_i32);

       return (CCPrepare) { .cond = TCG_COND_NE, .reg = reg,

                            .mask = -1, .no_setcond = true };

    }

}
