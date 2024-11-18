static void tcg_out_dat_rIN(TCGContext *s, int cond, int opc, int opneg,

                            TCGArg dst, TCGArg lhs, TCGArg rhs,

                            bool rhs_is_const)

{

    /* Emit either the reg,imm or reg,reg form of a data-processing insn.

     * rhs must satisfy the "rIN" constraint.

     */

    if (rhs_is_const) {

        int rot = encode_imm(rhs);

        if (rot < 0) {

            rhs = -rhs;

            rot = encode_imm(rhs);

            assert(rot >= 0);

            opc = opneg;

        }

        tcg_out_dat_imm(s, cond, opc, dst, lhs, rotl(rhs, rot) | (rot << 7));

    } else {

        tcg_out_dat_reg(s, cond, opc, dst, lhs, rhs, SHIFT_IMM_LSL(0));

    }

}
