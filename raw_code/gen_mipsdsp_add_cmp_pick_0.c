static void gen_mipsdsp_add_cmp_pick(DisasContext *ctx,

                                     uint32_t op1, uint32_t op2,

                                     int ret, int v1, int v2, int check_ret)

{

    const char *opn = "mipsdsp add compare pick";

    TCGv_i32 t0;

    TCGv t1;

    TCGv v1_t;

    TCGv v2_t;



    if ((ret == 0) && (check_ret == 1)) {

        /* Treat as NOP. */

        MIPS_DEBUG("NOP");

        return;

    }



    t0 = tcg_temp_new_i32();

    t1 = tcg_temp_new();

    v1_t = tcg_temp_new();

    v2_t = tcg_temp_new();



    gen_load_gpr(v1_t, v1);

    gen_load_gpr(v2_t, v2);



    switch (op1) {

    case OPC_APPEND_DSP:

        switch (op2) {

        case OPC_APPEND:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_append(cpu_gpr[ret], cpu_gpr[ret], v1_t, t0);

            break;

        case OPC_PREPEND:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_prepend(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        case OPC_BALIGN:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_balign(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        default:            /* Invid */

            MIPS_INVAL("MASK APPEND");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_CMPU_EQ_QB_DSP:

        switch (op2) {

        case OPC_CMPU_EQ_QB:

            check_dsp(ctx);

            gen_helper_cmpu_eq_qb(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPU_LT_QB:

            check_dsp(ctx);

            gen_helper_cmpu_lt_qb(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPU_LE_QB:

            check_dsp(ctx);

            gen_helper_cmpu_le_qb(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPGU_EQ_QB:

            check_dsp(ctx);

            gen_helper_cmpgu_eq_qb(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPGU_LT_QB:

            check_dsp(ctx);

            gen_helper_cmpgu_lt_qb(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPGU_LE_QB:

            check_dsp(ctx);

            gen_helper_cmpgu_le_qb(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPGDU_EQ_QB:

            check_dspr2(ctx);

            gen_helper_cmpgu_eq_qb(t1, v1_t, v2_t);

            tcg_gen_mov_tl(cpu_gpr[ret], t1);

            tcg_gen_andi_tl(cpu_dspctrl, cpu_dspctrl, 0xF0FFFFFF);

            tcg_gen_shli_tl(t1, t1, 24);

            tcg_gen_or_tl(cpu_dspctrl, cpu_dspctrl, t1);

            break;

        case OPC_CMPGDU_LT_QB:

            check_dspr2(ctx);

            gen_helper_cmpgu_lt_qb(t1, v1_t, v2_t);

            tcg_gen_mov_tl(cpu_gpr[ret], t1);

            tcg_gen_andi_tl(cpu_dspctrl, cpu_dspctrl, 0xF0FFFFFF);

            tcg_gen_shli_tl(t1, t1, 24);

            tcg_gen_or_tl(cpu_dspctrl, cpu_dspctrl, t1);

            break;

        case OPC_CMPGDU_LE_QB:

            check_dspr2(ctx);

            gen_helper_cmpgu_le_qb(t1, v1_t, v2_t);

            tcg_gen_mov_tl(cpu_gpr[ret], t1);

            tcg_gen_andi_tl(cpu_dspctrl, cpu_dspctrl, 0xF0FFFFFF);

            tcg_gen_shli_tl(t1, t1, 24);

            tcg_gen_or_tl(cpu_dspctrl, cpu_dspctrl, t1);

            break;

        case OPC_CMP_EQ_PH:

            check_dsp(ctx);

            gen_helper_cmp_eq_ph(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LT_PH:

            check_dsp(ctx);

            gen_helper_cmp_lt_ph(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LE_PH:

            check_dsp(ctx);

            gen_helper_cmp_le_ph(v1_t, v2_t, cpu_env);

            break;

        case OPC_PICK_QB:

            check_dsp(ctx);

            gen_helper_pick_qb(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_PICK_PH:

            check_dsp(ctx);

            gen_helper_pick_ph(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_PACKRL_PH:

            check_dsp(ctx);

            gen_helper_packrl_ph(cpu_gpr[ret], v1_t, v2_t);

            break;

        }

        break;

#ifdef TARGET_MIPS64

    case OPC_CMPU_EQ_OB_DSP:

        switch (op2) {

        case OPC_CMP_EQ_PW:

            check_dsp(ctx);

            gen_helper_cmp_eq_pw(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LT_PW:

            check_dsp(ctx);

            gen_helper_cmp_lt_pw(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LE_PW:

            check_dsp(ctx);

            gen_helper_cmp_le_pw(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_EQ_QH:

            check_dsp(ctx);

            gen_helper_cmp_eq_qh(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LT_QH:

            check_dsp(ctx);

            gen_helper_cmp_lt_qh(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMP_LE_QH:

            check_dsp(ctx);

            gen_helper_cmp_le_qh(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPGDU_EQ_OB:

            check_dspr2(ctx);

            gen_helper_cmpgdu_eq_ob(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPGDU_LT_OB:

            check_dspr2(ctx);

            gen_helper_cmpgdu_lt_ob(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPGDU_LE_OB:

            check_dspr2(ctx);

            gen_helper_cmpgdu_le_ob(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPGU_EQ_OB:

            check_dsp(ctx);

            gen_helper_cmpgu_eq_ob(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPGU_LT_OB:

            check_dsp(ctx);

            gen_helper_cmpgu_lt_ob(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPGU_LE_OB:

            check_dsp(ctx);

            gen_helper_cmpgu_le_ob(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_CMPU_EQ_OB:

            check_dsp(ctx);

            gen_helper_cmpu_eq_ob(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPU_LT_OB:

            check_dsp(ctx);

            gen_helper_cmpu_lt_ob(v1_t, v2_t, cpu_env);

            break;

        case OPC_CMPU_LE_OB:

            check_dsp(ctx);

            gen_helper_cmpu_le_ob(v1_t, v2_t, cpu_env);

            break;

        case OPC_PACKRL_PW:

            check_dsp(ctx);

            gen_helper_packrl_pw(cpu_gpr[ret], v1_t, v2_t);

            break;

        case OPC_PICK_OB:

            check_dsp(ctx);

            gen_helper_pick_ob(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_PICK_PW:

            check_dsp(ctx);

            gen_helper_pick_pw(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        case OPC_PICK_QH:

            check_dsp(ctx);

            gen_helper_pick_qh(cpu_gpr[ret], v1_t, v2_t, cpu_env);

            break;

        }

        break;

    case OPC_DAPPEND_DSP:

        switch (op2) {

        case OPC_DAPPEND:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_dappend(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        case OPC_PREPENDD:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_prependd(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        case OPC_PREPENDW:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_prependw(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        case OPC_DBALIGN:

            tcg_gen_movi_i32(t0, v2);

            gen_helper_dbalign(cpu_gpr[ret], v1_t, cpu_gpr[ret], t0);

            break;

        default:            /* Invalid */

            MIPS_INVAL("MASK DAPPEND");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

#endif

    }



    tcg_temp_free_i32(t0);

    tcg_temp_free(t1);

    tcg_temp_free(v1_t);

    tcg_temp_free(v2_t);



    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s", opn);

}
