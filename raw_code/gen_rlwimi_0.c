static void gen_rlwimi(DisasContext *ctx)

{

    uint32_t mb, me, sh;



    mb = MB(ctx->opcode);

    me = ME(ctx->opcode);

    sh = SH(ctx->opcode);

    if (likely(sh == (31-me) && mb <= me)) {

        tcg_gen_deposit_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rA(ctx->opcode)],

                           cpu_gpr[rS(ctx->opcode)], sh, me - mb + 1);

    } else {

        target_ulong mask;

        TCGv t1;

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

        mask = MASK(mb, me);

        t1 = tcg_temp_new();

        tcg_gen_andi_tl(t0, t0, mask);

        tcg_gen_andi_tl(t1, cpu_gpr[rA(ctx->opcode)], ~mask);

        tcg_gen_or_tl(cpu_gpr[rA(ctx->opcode)], t0, t1);

        tcg_temp_free(t0);

        tcg_temp_free(t1);

    }

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
