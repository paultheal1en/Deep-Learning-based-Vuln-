static void tgen_ext16s(TCGContext *s, TCGType type, TCGReg dest, TCGReg src)

{

    if (facilities & FACILITY_EXT_IMM) {

        tcg_out_insn(s, RRE, LGHR, dest, src);

        return;

    }



    if (type == TCG_TYPE_I32) {

        if (dest == src) {

            tcg_out_sh32(s, RS_SLL, dest, TCG_REG_NONE, 16);

        } else {

            tcg_out_sh64(s, RSY_SLLG, dest, src, TCG_REG_NONE, 16);

        }

        tcg_out_sh32(s, RS_SRA, dest, TCG_REG_NONE, 16);

    } else {

        tcg_out_sh64(s, RSY_SLLG, dest, src, TCG_REG_NONE, 48);

        tcg_out_sh64(s, RSY_SRAG, dest, dest, TCG_REG_NONE, 48);

    }

}
