static always_inline void gen_load_mem (DisasContext *ctx,

                                        void (*tcg_gen_qemu_load)(TCGv t0, TCGv t1, int flags),

                                        int ra, int rb, int32_t disp16,

                                        int fp, int clear)

{

    TCGv addr;



    if (unlikely(ra == 31))

        return;



    addr = tcg_temp_new(TCG_TYPE_I64);

    if (rb != 31) {

        tcg_gen_addi_i64(addr, cpu_ir[rb], disp16);

        if (clear)

            tcg_gen_andi_i64(addr, addr, ~0x7);

    } else {

        if (clear)

            disp16 &= ~0x7;

        tcg_gen_movi_i64(addr, disp16);

    }

    if (fp)

        tcg_gen_qemu_load(cpu_fir[ra], addr, ctx->mem_idx);

    else

        tcg_gen_qemu_load(cpu_ir[ra], addr, ctx->mem_idx);

    tcg_temp_free(addr);

}
