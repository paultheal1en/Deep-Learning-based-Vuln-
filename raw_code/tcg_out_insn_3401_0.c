static void tcg_out_insn_3401(TCGContext *s, AArch64Insn insn, TCGType ext,

                              TCGReg rd, TCGReg rn, uint64_t aimm)

{

    if (aimm > 0xfff) {

        assert((aimm & 0xfff) == 0);

        aimm >>= 12;

        assert(aimm <= 0xfff);

        aimm |= 1 << 12;  /* apply LSL 12 */

    }

    tcg_out32(s, insn | ext << 31 | aimm << 10 | rn << 5 | rd);

}
