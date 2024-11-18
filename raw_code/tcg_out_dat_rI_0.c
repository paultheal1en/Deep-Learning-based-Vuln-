static inline void tcg_out_dat_rI(TCGContext *s, int cond, int opc, TCGArg dst,

                                  TCGArg lhs, TCGArg rhs, int rhs_is_const)

{

    /* Emit either the reg,imm or reg,reg form of a data-processing insn.

     * rhs must satisfy the "rI" constraint.

     */

    if (rhs_is_const) {

        int rot = encode_imm(rhs);

        assert(rot >= 0);

        tcg_out_dat_imm(s, cond, opc, dst, lhs, rotl(rhs, rot) | (rot << 7));

    } else {

        tcg_out_dat_reg(s, cond, opc, dst, lhs, rhs, SHIFT_IMM_LSL(0));

    }

}
