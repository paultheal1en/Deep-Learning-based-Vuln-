static void tgen_compare_imm_branch(TCGContext *s, S390Opcode opc, int cc,

                                    TCGReg r1, int i2, int labelno)

{

    TCGLabel* l = &s->labels[labelno];

    tcg_target_long off;



    if (l->has_value) {

        off = l->u.value_ptr - s->code_ptr;

    } else {

        /* We need to keep the offset unchanged for retranslation.  */

        off = s->code_ptr[1];

        tcg_out_reloc(s, s->code_ptr + 1, R_390_PC16DBL, labelno, -2);

    }



    tcg_out16(s, (opc & 0xff00) | (r1 << 4) | cc);

    tcg_out16(s, off);

    tcg_out16(s, (i2 << 8) | (opc & 0xff));

}
