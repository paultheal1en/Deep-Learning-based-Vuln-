static void tcg_out_movi(TCGContext *s, TCGType type, TCGReg rd,

                         tcg_target_long value)

{

    AArch64Insn insn;



    if (type == TCG_TYPE_I32) {

        value = (uint32_t)value;

    }



    /* count trailing zeros in 16 bit steps, mapping 64 to 0. Emit the

       first MOVZ with the half-word immediate skipping the zeros, with a shift

       (LSL) equal to this number. Then all next instructions use MOVKs.

       Zero the processed half-word in the value, continue until empty.

       We build the final result 16bits at a time with up to 4 instructions,

       but do not emit instructions for 16bit zero holes. */

    insn = I3405_MOVZ;

    do {

        unsigned shift = ctz64(value) & (63 & -16);

        tcg_out_insn_3405(s, insn, shift >= 32, rd, value >> shift, shift);

        value &= ~(0xffffUL << shift);

        insn = I3405_MOVK;

    } while (value);

}
