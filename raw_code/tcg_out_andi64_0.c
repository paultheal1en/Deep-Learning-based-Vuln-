static void tcg_out_andi64(TCGContext *s, TCGReg dst, TCGReg src, uint64_t c)

{

    int mb, me;



    assert(TCG_TARGET_REG_BITS == 64);

    if (mask64_operand(c, &mb, &me)) {

        if (mb == 0) {

            tcg_out_rld(s, RLDICR, dst, src, 0, me);

        } else {

            tcg_out_rld(s, RLDICL, dst, src, 0, mb);

        }

    } else if ((c & 0xffff) == c) {

        tcg_out32(s, ANDI | SAI(src, dst, c));

        return;

    } else if ((c & 0xffff0000) == c) {

        tcg_out32(s, ANDIS | SAI(src, dst, c >> 16));

        return;

    } else {

        tcg_out_movi(s, TCG_TYPE_I64, TCG_REG_R0, c);

        tcg_out32(s, AND | SAB(src, dst, TCG_REG_R0));

    }

}
