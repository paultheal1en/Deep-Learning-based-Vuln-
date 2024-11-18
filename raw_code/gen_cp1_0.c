static void gen_cp1 (DisasContext *ctx, uint32_t opc, int rt, int fs)

{

    const char *opn = "cp1 move";

    TCGv t0 = tcg_temp_new();



    switch (opc) {

    case OPC_MFC1:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            tcg_gen_ext_i32_tl(t0, fp0);

            tcg_temp_free_i32(fp0);

        }

        gen_store_gpr(t0, rt);

        opn = "mfc1";

        break;

    case OPC_MTC1:

        gen_load_gpr(t0, rt);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            tcg_gen_trunc_tl_i32(fp0, t0);

            gen_store_fpr32(fp0, fs);

            tcg_temp_free_i32(fp0);

        }

        opn = "mtc1";

        break;

    case OPC_CFC1:

        gen_helper_1e0i(cfc1, t0, fs);

        gen_store_gpr(t0, rt);

        opn = "cfc1";

        break;

    case OPC_CTC1:

        gen_load_gpr(t0, rt);

        {

            TCGv_i32 fs_tmp = tcg_const_i32(fs);



            gen_helper_0e2i(ctc1, t0, fs_tmp, rt);

            tcg_temp_free_i32(fs_tmp);

        }

        opn = "ctc1";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DMFC1:

        gen_load_fpr64(ctx, t0, fs);

        gen_store_gpr(t0, rt);

        opn = "dmfc1";

        break;

    case OPC_DMTC1:

        gen_load_gpr(t0, rt);

        gen_store_fpr64(ctx, t0, fs);

        opn = "dmtc1";

        break;

#endif

    case OPC_MFHC1:

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            gen_load_fpr32h(fp0, fs);

            tcg_gen_ext_i32_tl(t0, fp0);

            tcg_temp_free_i32(fp0);

        }

        gen_store_gpr(t0, rt);

        opn = "mfhc1";

        break;

    case OPC_MTHC1:

        gen_load_gpr(t0, rt);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();



            tcg_gen_trunc_tl_i32(fp0, t0);

            gen_store_fpr32h(fp0, fs);

            tcg_temp_free_i32(fp0);

        }

        opn = "mthc1";

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        goto out;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s %s", opn, regnames[rt], fregnames[fs]);



 out:

    tcg_temp_free(t0);

}
