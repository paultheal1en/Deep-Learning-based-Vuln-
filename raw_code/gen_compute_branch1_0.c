static void gen_compute_branch1 (CPUState *env, DisasContext *ctx, uint32_t op,

                                 int32_t cc, int32_t offset)

{

    target_ulong btarget;

    const char *opn = "cp1 cond branch";

    TCGv t0 = tcg_temp_local_new(TCG_TYPE_TL);

    TCGv t1 = tcg_temp_local_new(TCG_TYPE_TL);



    if (cc != 0)

        check_insn(env, ctx, ISA_MIPS4 | ISA_MIPS32);



    btarget = ctx->pc + 4 + offset;



    switch (op) {

    case OPC_BC1F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_not_tl(t0, t0);

            tcg_gen_movi_tl(t1, 0x1 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1f";

        goto not_likely;

    case OPC_BC1FL:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_not_tl(t0, t0);

            tcg_gen_movi_tl(t1, 0x1 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1fl";

        goto likely;

    case OPC_BC1T:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_movi_tl(t1, 0x1 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1t";

        goto not_likely;

    case OPC_BC1TL:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_movi_tl(t1, 0x1 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1tl";

    likely:

        ctx->hflags |= MIPS_HFLAG_BL;

        tcg_gen_trunc_tl_i32(bcond, t0);

        break;

    case OPC_BC1FANY2:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_not_tl(t0, t0);

            tcg_gen_movi_tl(t1, 0x3 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1any2f";

        goto not_likely;

    case OPC_BC1TANY2:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_movi_tl(t1, 0x3 << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1any2t";

        goto not_likely;

    case OPC_BC1FANY4:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_not_tl(t0, t0);

            tcg_gen_movi_tl(t1, 0xf << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1any4f";

        goto not_likely;

    case OPC_BC1TANY4:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



            get_fp_cond(r_tmp1);

            tcg_gen_ext_i32_tl(t0, r_tmp1);

            tcg_temp_free(r_tmp1);

            tcg_gen_movi_tl(t1, 0xf << cc);

            tcg_gen_and_tl(t0, t0, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            tcg_gen_movi_tl(t0, 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(t0, 1);

            gen_set_label(l2);

        }

        opn = "bc1any4t";

    not_likely:

        ctx->hflags |= MIPS_HFLAG_BC;

        tcg_gen_trunc_tl_i32(bcond, t0);

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        goto out;

    }

    MIPS_DEBUG("%s: cond %02x target " TARGET_FMT_lx, opn,

               ctx->hflags, btarget);

    ctx->btarget = btarget;



 out:

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
