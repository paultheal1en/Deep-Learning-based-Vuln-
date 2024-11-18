static void gen_flt3_arith (DisasContext *ctx, uint32_t opc,

                            int fd, int fr, int fs, int ft)

{

    const char *opn = "flt3_arith";



    switch (opc) {

    case OPC_ALNV_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv_i32 fp = tcg_temp_new_i32();

            TCGv_i32 fph = tcg_temp_new_i32();

            int l1 = gen_new_label();

            int l2 = gen_new_label();



            gen_load_gpr(t0, fr);

            tcg_gen_andi_tl(t0, t0, 0x7);



            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0, l1);

            gen_load_fpr32(fp, fs);

            gen_load_fpr32h(ctx, fph, fs);

            gen_store_fpr32(fp, fd);

            gen_store_fpr32h(ctx, fph, fd);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, 4, l2);

            tcg_temp_free(t0);

#ifdef TARGET_WORDS_BIGENDIAN

            gen_load_fpr32(fp, fs);

            gen_load_fpr32h(ctx, fph, ft);

            gen_store_fpr32h(ctx, fp, fd);

            gen_store_fpr32(fph, fd);

#else

            gen_load_fpr32h(ctx, fph, fs);

            gen_load_fpr32(fp, ft);

            gen_store_fpr32(fph, fd);

            gen_store_fpr32h(ctx, fp, fd);

#endif

            gen_set_label(l2);

            tcg_temp_free_i32(fp);

            tcg_temp_free_i32(fph);

        }

        opn = "alnv.ps";

        break;

    case OPC_MADD_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();

            TCGv_i32 fp2 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_load_fpr32(fp2, fr);

            gen_helper_float_madd_s(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp2, fd);

            tcg_temp_free_i32(fp2);

        }

        opn = "madd.s";

        break;

    case OPC_MADD_D:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fd | fs | ft | fr);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_madd_d(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "madd.d";

        break;

    case OPC_MADD_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_madd_ps(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "madd.ps";

        break;

    case OPC_MSUB_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();

            TCGv_i32 fp2 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_load_fpr32(fp2, fr);

            gen_helper_float_msub_s(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp2, fd);

            tcg_temp_free_i32(fp2);

        }

        opn = "msub.s";

        break;

    case OPC_MSUB_D:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fd | fs | ft | fr);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_msub_d(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "msub.d";

        break;

    case OPC_MSUB_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_msub_ps(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "msub.ps";

        break;

    case OPC_NMADD_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();

            TCGv_i32 fp2 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_load_fpr32(fp2, fr);

            gen_helper_float_nmadd_s(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp2, fd);

            tcg_temp_free_i32(fp2);

        }

        opn = "nmadd.s";

        break;

    case OPC_NMADD_D:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fd | fs | ft | fr);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_nmadd_d(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "nmadd.d";

        break;

    case OPC_NMADD_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_nmadd_ps(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "nmadd.ps";

        break;

    case OPC_NMSUB_S:

        check_cop1x(ctx);

        {

            TCGv_i32 fp0 = tcg_temp_new_i32();

            TCGv_i32 fp1 = tcg_temp_new_i32();

            TCGv_i32 fp2 = tcg_temp_new_i32();



            gen_load_fpr32(fp0, fs);

            gen_load_fpr32(fp1, ft);

            gen_load_fpr32(fp2, fr);

            gen_helper_float_nmsub_s(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i32(fp0);

            tcg_temp_free_i32(fp1);

            gen_store_fpr32(fp2, fd);

            tcg_temp_free_i32(fp2);

        }

        opn = "nmsub.s";

        break;

    case OPC_NMSUB_D:

        check_cop1x(ctx);

        check_cp1_registers(ctx, fd | fs | ft | fr);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_nmsub_d(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "nmsub.d";

        break;

    case OPC_NMSUB_PS:

        check_cp1_64bitmode(ctx);

        {

            TCGv_i64 fp0 = tcg_temp_new_i64();

            TCGv_i64 fp1 = tcg_temp_new_i64();

            TCGv_i64 fp2 = tcg_temp_new_i64();



            gen_load_fpr64(ctx, fp0, fs);

            gen_load_fpr64(ctx, fp1, ft);

            gen_load_fpr64(ctx, fp2, fr);

            gen_helper_float_nmsub_ps(fp2, cpu_env, fp0, fp1, fp2);

            tcg_temp_free_i64(fp0);

            tcg_temp_free_i64(fp1);

            gen_store_fpr64(ctx, fp2, fd);

            tcg_temp_free_i64(fp2);

        }

        opn = "nmsub.ps";

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        return;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %s, %s, %s", opn, fregnames[fd], fregnames[fr],

               fregnames[fs], fregnames[ft]);

}
