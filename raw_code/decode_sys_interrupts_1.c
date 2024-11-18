static void decode_sys_interrupts(CPUTriCoreState *env, DisasContext *ctx)

{

    uint32_t op2;

    uint32_t r1;

    TCGLabel *l1;

    TCGv tmp;



    op2 = MASK_OP_SYS_OP2(ctx->opcode);

    r1  = MASK_OP_SYS_S1D(ctx->opcode);



    switch (op2) {

    case OPC2_32_SYS_DEBUG:

        /* raise EXCP_DEBUG */

        break;

    case OPC2_32_SYS_DISABLE:

        tcg_gen_andi_tl(cpu_ICR, cpu_ICR, ~MASK_ICR_IE);

        break;

    case OPC2_32_SYS_DSYNC:

        break;

    case OPC2_32_SYS_ENABLE:

        tcg_gen_ori_tl(cpu_ICR, cpu_ICR, MASK_ICR_IE);

        break;

    case OPC2_32_SYS_ISYNC:

        break;

    case OPC2_32_SYS_NOP:

        break;

    case OPC2_32_SYS_RET:

        gen_compute_branch(ctx, op2, 0, 0, 0, 0);

        break;

    case OPC2_32_SYS_FRET:

        gen_fret(ctx);

        break;

    case OPC2_32_SYS_RFE:

        gen_helper_rfe(cpu_env);

        tcg_gen_exit_tb(0);

        ctx->bstate = BS_BRANCH;

        break;

    case OPC2_32_SYS_RFM:

        if ((ctx->hflags & TRICORE_HFLAG_KUU) == TRICORE_HFLAG_SM) {

            tmp = tcg_temp_new();

            l1 = gen_new_label();



            tcg_gen_ld32u_tl(tmp, cpu_env, offsetof(CPUTriCoreState, DBGSR));

            tcg_gen_andi_tl(tmp, tmp, MASK_DBGSR_DE);

            tcg_gen_brcondi_tl(TCG_COND_NE, tmp, 1, l1);

            gen_helper_rfm(cpu_env);

            gen_set_label(l1);

            tcg_gen_exit_tb(0);

            ctx->bstate = BS_BRANCH;

            tcg_temp_free(tmp);

        } else {

            /* generate privilege trap */

        }

        break;

    case OPC2_32_SYS_RSLCX:

        gen_helper_rslcx(cpu_env);

        break;

    case OPC2_32_SYS_SVLCX:

        gen_helper_svlcx(cpu_env);

        break;

    case OPC2_32_SYS_RESTORE:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            if ((ctx->hflags & TRICORE_HFLAG_KUU) == TRICORE_HFLAG_SM ||

                (ctx->hflags & TRICORE_HFLAG_KUU) == TRICORE_HFLAG_UM1) {

                tcg_gen_deposit_tl(cpu_ICR, cpu_ICR, cpu_gpr_d[r1], 8, 1);

            } /* else raise privilege trap */

        } /* else raise illegal opcode trap */

        break;

    case OPC2_32_SYS_TRAPSV:

        l1 = gen_new_label();

        tcg_gen_brcondi_tl(TCG_COND_GE, cpu_PSW_SV, 0, l1);

        generate_trap(ctx, TRAPC_ASSERT, TIN5_SOVF);

        gen_set_label(l1);

        break;

    case OPC2_32_SYS_TRAPV:

        l1 = gen_new_label();

        tcg_gen_brcondi_tl(TCG_COND_GE, cpu_PSW_V, 0, l1);

        generate_trap(ctx, TRAPC_ASSERT, TIN5_OVF);

        gen_set_label(l1);

        break;

    }

}
