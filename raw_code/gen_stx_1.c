static void gen_stx(DisasContext *dc, uint32_t code, uint32_t flags)

{

    I_TYPE(instr, code);

    TCGv val = load_gpr(dc, instr.b);



    TCGv addr = tcg_temp_new();

    tcg_gen_addi_tl(addr, load_gpr(dc, instr.a), instr.imm16s);

    tcg_gen_qemu_st_tl(val, addr, dc->mem_idx, flags);

    tcg_temp_free(addr);

}
