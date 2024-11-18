static void tcg_out_brcond_i64(TCGContext *s, TCGCond cond, TCGReg arg1,

                               int32_t arg2, int const_arg2, int label)

{

    /* For 64-bit signed comparisons vs zero, we can avoid the compare.  */

    if (arg2 == 0 && !is_unsigned_cond(cond)) {

        TCGLabel *l = &s->labels[label];

        int off16;



        if (l->has_value) {

            off16 = INSN_OFF16(tcg_pcrel_diff(s, l->u.value_ptr));

        } else {

            /* Make sure to preserve destinations during retranslation.  */

            off16 = *s->code_ptr & INSN_OFF16(-1);

            tcg_out_reloc(s, s->code_ptr, R_SPARC_WDISP16, label, 0);

        }

        tcg_out32(s, INSN_OP(0) | INSN_OP2(3) | BPR_PT | INSN_RS1(arg1)

                  | INSN_COND(tcg_cond_to_rcond[cond]) | off16);

    } else {

        tcg_out_cmp(s, arg1, arg2, const_arg2);

        tcg_out_bpcc(s, tcg_cond_to_bcond[cond], BPCC_XCC | BPCC_PT, label);

    }

    tcg_out_nop(s);

}
