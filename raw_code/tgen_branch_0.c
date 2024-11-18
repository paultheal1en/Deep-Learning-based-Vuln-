static void tgen_branch(TCGContext *s, int cc, int labelno)

{

    TCGLabel* l = &s->labels[labelno];

    if (l->has_value) {

        tgen_gotoi(s, cc, l->u.value_ptr);

    } else if (USE_LONG_BRANCHES) {

        tcg_out16(s, RIL_BRCL | (cc << 4));

        tcg_out_reloc(s, s->code_ptr, R_390_PC32DBL, labelno, -2);

        s->code_ptr += 2;

    } else {

        tcg_out16(s, RI_BRC | (cc << 4));

        tcg_out_reloc(s, s->code_ptr, R_390_PC16DBL, labelno, -2);

        s->code_ptr += 1;

    }

}
