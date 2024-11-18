static void gen_muldiv (DisasContext *ctx, uint32_t opc,

                        int rs, int rt)

{

    const char *opn = "mul/div";

    TCGv t0, t1;

    unsigned int acc;



    switch (opc) {

    case OPC_DIV:

    case OPC_DIVU:

#if defined(TARGET_MIPS64)

    case OPC_DDIV:

    case OPC_DDIVU:

#endif

        t0 = tcg_temp_local_new();

        t1 = tcg_temp_local_new();

        break;

    default:

        t0 = tcg_temp_new();

        t1 = tcg_temp_new();

        break;

    }



    gen_load_gpr(t0, rs);

    gen_load_gpr(t1, rt);

    switch (opc) {

    case OPC_DIV:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();



            tcg_gen_ext32s_tl(t0, t0);

            tcg_gen_ext32s_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, INT_MIN, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1, l2);



            tcg_gen_mov_tl(cpu_LO[0], t0);

            tcg_gen_movi_tl(cpu_HI[0], 0);

            tcg_gen_br(l1);

            gen_set_label(l2);

            tcg_gen_div_tl(cpu_LO[0], t0, t1);

            tcg_gen_rem_tl(cpu_HI[0], t0, t1);

            tcg_gen_ext32s_tl(cpu_LO[0], cpu_LO[0]);

            tcg_gen_ext32s_tl(cpu_HI[0], cpu_HI[0]);

            gen_set_label(l1);

        }

        opn = "div";

        break;

    case OPC_DIVU:

        {

            int l1 = gen_new_label();



            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_divu_tl(cpu_LO[0], t0, t1);

            tcg_gen_remu_tl(cpu_HI[0], t0, t1);

            tcg_gen_ext32s_tl(cpu_LO[0], cpu_LO[0]);

            tcg_gen_ext32s_tl(cpu_HI[0], cpu_HI[0]);

            gen_set_label(l1);

        }

        opn = "divu";

        break;

    case OPC_MULT:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext_tl_i64(t2, t0);

            tcg_gen_ext_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "mult";

        break;

    case OPC_MULTU:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_extu_tl_i64(t2, t0);

            tcg_gen_extu_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "multu";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DDIV:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();



            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, -1LL << 63, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1LL, l2);

            tcg_gen_mov_tl(cpu_LO[0], t0);

            tcg_gen_movi_tl(cpu_HI[0], 0);

            tcg_gen_br(l1);

            gen_set_label(l2);

            tcg_gen_div_i64(cpu_LO[0], t0, t1);

            tcg_gen_rem_i64(cpu_HI[0], t0, t1);

            gen_set_label(l1);

        }

        opn = "ddiv";

        break;

    case OPC_DDIVU:

        {

            int l1 = gen_new_label();



            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_divu_i64(cpu_LO[0], t0, t1);

            tcg_gen_remu_i64(cpu_HI[0], t0, t1);

            gen_set_label(l1);

        }

        opn = "ddivu";

        break;

    case OPC_DMULT:

        gen_helper_dmult(cpu_env, t0, t1);

        opn = "dmult";

        break;

    case OPC_DMULTU:

        gen_helper_dmultu(cpu_env, t0, t1);

        opn = "dmultu";

        break;

#endif

    case OPC_MADD:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext_tl_i64(t2, t0);

            tcg_gen_ext_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_gen_concat_tl_i64(t3, cpu_LO[acc], cpu_HI[acc]);

            tcg_gen_add_i64(t2, t2, t3);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "madd";

        break;

    case OPC_MADDU:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_extu_tl_i64(t2, t0);

            tcg_gen_extu_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_gen_concat_tl_i64(t3, cpu_LO[acc], cpu_HI[acc]);

            tcg_gen_add_i64(t2, t2, t3);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "maddu";

        break;

    case OPC_MSUB:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext_tl_i64(t2, t0);

            tcg_gen_ext_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_gen_concat_tl_i64(t3, cpu_LO[acc], cpu_HI[acc]);

            tcg_gen_sub_i64(t2, t3, t2);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "msub";

        break;

    case OPC_MSUBU:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i64 t3 = tcg_temp_new_i64();

            acc = ((ctx->opcode) >> 11) & 0x03;

            if (acc != 0) {

                check_dsp(ctx);

            }



            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_extu_tl_i64(t2, t0);

            tcg_gen_extu_tl_i64(t3, t1);

            tcg_gen_mul_i64(t2, t2, t3);

            tcg_gen_concat_tl_i64(t3, cpu_LO[acc], cpu_HI[acc]);

            tcg_gen_sub_i64(t2, t3, t2);

            tcg_temp_free_i64(t3);

            tcg_gen_trunc_i64_tl(t0, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_tl(t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_ext32s_tl(cpu_LO[acc], t0);

            tcg_gen_ext32s_tl(cpu_HI[acc], t1);

        }

        opn = "msubu";

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        goto out;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s %s", opn, regnames[rs], regnames[rt]);

 out:

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
