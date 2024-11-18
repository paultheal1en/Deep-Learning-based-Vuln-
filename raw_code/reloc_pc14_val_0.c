static uint16_t reloc_pc14_val(tcg_insn_unit *pc, tcg_insn_unit *target)

{

    ptrdiff_t disp = tcg_ptr_byte_diff(target, pc);

    assert(disp == (int16_t) disp);

    return disp & 0xfffc;

}
