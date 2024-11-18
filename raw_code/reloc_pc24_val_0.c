static uint32_t reloc_pc24_val(tcg_insn_unit *pc, tcg_insn_unit *target)

{

    ptrdiff_t disp = tcg_ptr_byte_diff(target, pc);

    assert(in_range_b(disp));

    return disp & 0x3fffffc;

}
