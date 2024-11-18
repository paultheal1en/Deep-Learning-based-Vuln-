static void tci_out_label(TCGContext *s, TCGArg arg)

{

    TCGLabel *label = &s->labels[arg];

    if (label->has_value) {

        tcg_out_i(s, label->u.value);

        assert(label->u.value);

    } else {

        tcg_out_reloc(s, s->code_ptr, sizeof(tcg_target_ulong), arg, 0);

        tcg_out_i(s, 0);

    }

}
