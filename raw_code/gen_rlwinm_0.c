static void gen_rlwinm(DisasContext *ctx)

{

    uint32_t mb, me, sh;



    sh = SH(ctx->opcode);

    mb = MB(ctx->opcode);

    me = ME(ctx->opcode);



    if (likely(mb == 0 && me == (31 - sh))) {

        if (likely(sh == 0)) {

            tcg_gen_ext32u_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rS(ctx->opcode)]);

        } else {

            TCGv t0 = tcg_temp_new();

            tcg_gen_ext32u_tl(t0, cpu_gpr[rS(ctx->opcode)]);

            tcg_gen_shli_tl(t0, t0, sh);

            tcg_gen_ext32u_tl(cpu_gpr[rA(ctx->opcode)], t0);

            tcg_temp_free(t0);

        }

    } else if (likely(sh != 0 && me == 31 && sh == (32 - mb))) {

        TCGv t0 = tcg_temp_new();

        tcg_gen_ext32u_tl(t0, cpu_gpr[rS(ctx->opcode)]);

        tcg_gen_shri_tl(t0, t0, mb);

        tcg_gen_ext32u_tl(cpu_gpr[rA(ctx->opcode)], t0);

        tcg_temp_free(t0);

    } else if (likely(mb == 0 && me == 31)) {

        TCGv_i32 t0 = tcg_temp_new_i32();

        tcg_gen_trunc_tl_i32(t0, cpu_gpr[rS(ctx->opcode)]);

        tcg_gen_rotli_i32(t0, t0, sh);

        tcg_gen_extu_i32_tl(cpu_gpr[rA(ctx->opcode)], t0);

        tcg_temp_free_i32(t0);

    } else {

        TCGv t0 = tcg_temp_new();

#if defined(TARGET_PPC64)

        tcg_gen_deposit_i64(t0, cpu_gpr[rS(ctx->opcode)],

            cpu_gpr[rS(ctx->opcode)], 32, 32);

        tcg_gen_rotli_i64(t0, t0, sh);

#else

        tcg_gen_rotli_i32(t0, cpu_gpr[rS(ctx->opcode)], sh);

#endif

#if defined(TARGET_PPC64)

        mb += 32;

        me += 32;

#endif

        tcg_gen_andi_tl(cpu_gpr[rA(ctx->opcode)], t0, MASK(mb, me));

        tcg_temp_free(t0);

    }

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
