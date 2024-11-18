static void gen_ldx(DisasContext *dc, uint32_t code, uint32_t flags)

{

    I_TYPE(instr, code);



    TCGv addr = tcg_temp_new();

    TCGv data;



    /*

     * WARNING: Loads into R_ZERO are ignored, but we must generate the

     *          memory access itself to emulate the CPU precisely. Load

     *          from a protected page to R_ZERO will cause SIGSEGV on

     *          the Nios2 CPU.

     */

    if (likely(instr.b != R_ZERO)) {

        data = dc->cpu_R[instr.b];

    } else {

        data = tcg_temp_new();

    }



    tcg_gen_addi_tl(addr, load_gpr(dc, instr.a), instr.imm16s);

    tcg_gen_qemu_ld_tl(data, addr, dc->mem_idx, flags);



    if (unlikely(instr.b == R_ZERO)) {

        tcg_temp_free(data);

    }



    tcg_temp_free(addr);

}
