static void gen_ldst_pair (DisasContext *ctx, uint32_t opc, int rd,

                           int base, int16_t offset)

{

    const char *opn = "ldst_pair";

    TCGv t0, t1;



    if (ctx->hflags & MIPS_HFLAG_BMASK || rd == 31 || rd == base) {

        generate_exception(ctx, EXCP_RI);

        return;

    }



    t0 = tcg_temp_new();

    t1 = tcg_temp_new();



    gen_base_offset_addr(ctx, t0, base, offset);



    switch (opc) {

    case LWP:

        save_cpu_state(ctx, 0);

        op_ld_lw(t1, t0, ctx);

        gen_store_gpr(t1, rd);

        tcg_gen_movi_tl(t1, 4);

        gen_op_addr_add(ctx, t0, t0, t1);

        op_ld_lw(t1, t0, ctx);

        gen_store_gpr(t1, rd+1);

        opn = "lwp";

        break;

    case SWP:

        save_cpu_state(ctx, 0);

        gen_load_gpr(t1, rd);

        op_st_sw(t1, t0, ctx);

        tcg_gen_movi_tl(t1, 4);

        gen_op_addr_add(ctx, t0, t0, t1);

        gen_load_gpr(t1, rd+1);

        op_st_sw(t1, t0, ctx);

        opn = "swp";

        break;

#ifdef TARGET_MIPS64

    case LDP:

        save_cpu_state(ctx, 0);

        op_ld_ld(t1, t0, ctx);

        gen_store_gpr(t1, rd);

        tcg_gen_movi_tl(t1, 8);

        gen_op_addr_add(ctx, t0, t0, t1);

        op_ld_ld(t1, t0, ctx);

        gen_store_gpr(t1, rd+1);

        opn = "ldp";

        break;

    case SDP:

        save_cpu_state(ctx, 0);

        gen_load_gpr(t1, rd);

        op_st_sd(t1, t0, ctx);

        tcg_gen_movi_tl(t1, 8);

        gen_op_addr_add(ctx, t0, t0, t1);

        gen_load_gpr(t1, rd+1);

        op_st_sd(t1, t0, ctx);

        opn = "sdp";

        break;

#endif

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s, %s, %d(%s)", opn, regnames[rd], offset, regnames[base]);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
