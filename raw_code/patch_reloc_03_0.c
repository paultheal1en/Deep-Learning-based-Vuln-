static inline void patch_reloc(tcg_insn_unit *code_ptr, int type,

                               intptr_t value, intptr_t addend)

{

    assert(addend == 0);

    switch (type) {

    case R_AARCH64_JUMP26:

    case R_AARCH64_CALL26:

        reloc_pc26(code_ptr, (tcg_insn_unit *)value);

        break;

    case R_AARCH64_CONDBR19:

        reloc_pc19(code_ptr, (tcg_insn_unit *)value);

        break;

    default:

        tcg_abort();

    }

}
