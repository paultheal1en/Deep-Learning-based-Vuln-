static void gen_ldarx(DisasContext *ctx)

{

    TCGv t0;

    gen_set_access_type(ctx, ACCESS_RES);

    t0 = tcg_temp_local_new();

    gen_addr_reg_index(ctx, t0);

    gen_check_align(ctx, t0, 0x07);

    gen_qemu_ld64(ctx, cpu_gpr[rD(ctx->opcode)], t0);

    tcg_gen_mov_tl(cpu_reserve, t0);

    tcg_temp_free(t0);

}
