static inline uint32_t reloc_26_val(tcg_insn_unit *pc, tcg_insn_unit *target)

{

    assert((((uintptr_t)pc ^ (uintptr_t)target) & 0xf0000000) == 0);

    return ((uintptr_t)target >> 2) & 0x3ffffff;

}
