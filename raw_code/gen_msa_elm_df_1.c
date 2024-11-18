static void gen_msa_elm_df(CPUMIPSState *env, DisasContext *ctx, uint32_t df,

        uint32_t n)

{

#define MASK_MSA_ELM(op)    (MASK_MSA_MINOR(op) | (op & (0xf << 22)))

    uint8_t ws = (ctx->opcode >> 11) & 0x1f;

    uint8_t wd = (ctx->opcode >> 6) & 0x1f;



    TCGv_i32 tws = tcg_const_i32(ws);

    TCGv_i32 twd = tcg_const_i32(wd);

    TCGv_i32 tn  = tcg_const_i32(n);

    TCGv_i32 tdf = tcg_const_i32(df);



    switch (MASK_MSA_ELM(ctx->opcode)) {

    case OPC_SLDI_df:

        gen_helper_msa_sldi_df(cpu_env, tdf, twd, tws, tn);

        break;

    case OPC_SPLATI_df:

        gen_helper_msa_splati_df(cpu_env, tdf, twd, tws, tn);

        break;

    case OPC_INSVE_df:

        gen_helper_msa_insve_df(cpu_env, tdf, twd, tws, tn);

        break;

    case OPC_COPY_S_df:

    case OPC_COPY_U_df:

    case OPC_INSERT_df:

#if !defined(TARGET_MIPS64)

        /* Double format valid only for MIPS64 */

        if (df == DF_DOUBLE) {

            generate_exception_end(ctx, EXCP_RI);

            break;

        }

#endif

        switch (MASK_MSA_ELM(ctx->opcode)) {

        case OPC_COPY_S_df:

            gen_helper_msa_copy_s_df(cpu_env, tdf, twd, tws, tn);

            break;

        case OPC_COPY_U_df:

            gen_helper_msa_copy_u_df(cpu_env, tdf, twd, tws, tn);

            break;

        case OPC_INSERT_df:

            gen_helper_msa_insert_df(cpu_env, tdf, twd, tws, tn);

            break;

        }

        break;

    default:

        MIPS_INVAL("MSA instruction");

        generate_exception_end(ctx, EXCP_RI);

    }

    tcg_temp_free_i32(twd);

    tcg_temp_free_i32(tws);

    tcg_temp_free_i32(tn);

    tcg_temp_free_i32(tdf);

}
