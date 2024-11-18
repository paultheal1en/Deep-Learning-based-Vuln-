static void gen_std(DisasContext *ctx)

{

    int rs;

    TCGv EA;



    rs = rS(ctx->opcode);

    if ((ctx->opcode & 0x3) == 0x2) {

#if defined(CONFIG_USER_ONLY)

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

        /* stq */

        if (unlikely(ctx->mem_idx == 0)) {

            gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

            return;

        }

        if (unlikely(rs & 1)) {

            gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

            return;

        }

        if (unlikely(ctx->le_mode)) {

            /* Little-endian mode is not handled */

            gen_exception_err(ctx, POWERPC_EXCP_ALIGN, POWERPC_EXCP_ALIGN_LE);

            return;

        }

        gen_set_access_type(ctx, ACCESS_INT);

        EA = tcg_temp_new();

        gen_addr_imm_index(ctx, EA, 0x03);

        gen_qemu_st64(ctx, cpu_gpr[rs], EA);

        gen_addr_add(ctx, EA, EA, 8);

        gen_qemu_st64(ctx, cpu_gpr[rs+1], EA);

        tcg_temp_free(EA);

#endif

    } else {

        /* std / stdu */

        if (Rc(ctx->opcode)) {

            if (unlikely(rA(ctx->opcode) == 0)) {

                gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

                return;

            }

        }

        gen_set_access_type(ctx, ACCESS_INT);

        EA = tcg_temp_new();

        gen_addr_imm_index(ctx, EA, 0x03);

        gen_qemu_st64(ctx, cpu_gpr[rs], EA);

        if (Rc(ctx->opcode))

            tcg_gen_mov_tl(cpu_gpr[rA(ctx->opcode)], EA);

        tcg_temp_free(EA);

    }

}
