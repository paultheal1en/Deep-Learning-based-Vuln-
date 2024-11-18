static void tcg_out_bpcc(TCGContext *s, int scond, int flags, int label)

{

    TCGLabel *l = &s->labels[label];

    int off19;



    if (l->has_value) {

        off19 = INSN_OFF19(tcg_pcrel_diff(s, l->u.value_ptr));

    } else {

        /* Make sure to preserve destinations during retranslation.  */

        off19 = *s->code_ptr & INSN_OFF19(-1);

        tcg_out_reloc(s, s->code_ptr, R_SPARC_WDISP19, label, 0);

    }

    tcg_out_bpcc0(s, scond, flags, off19);

}
