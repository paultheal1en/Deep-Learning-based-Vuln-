void tcg_gen_brcond_i64(TCGCond cond, TCGv_i64 arg1, TCGv_i64 arg2, int label)

{

    if (cond == TCG_COND_ALWAYS) {

        tcg_gen_br(label);

    } else if (cond != TCG_COND_NEVER) {

        if (TCG_TARGET_REG_BITS == 32) {

            tcg_gen_op6ii_i32(INDEX_op_brcond2_i32, TCGV_LOW(arg1),

                              TCGV_HIGH(arg1), TCGV_LOW(arg2),

                              TCGV_HIGH(arg2), cond, label);

        } else {

            tcg_gen_op4ii_i64(INDEX_op_brcond_i64, arg1, arg2, cond, label);

        }

    }

}
