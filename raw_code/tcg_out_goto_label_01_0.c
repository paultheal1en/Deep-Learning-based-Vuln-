static inline void tcg_out_goto_label(TCGContext *s, int label_index)

{

    TCGLabel *l = &s->labels[label_index];



    if (!l->has_value) {

        tcg_out_reloc(s, s->code_ptr, R_AARCH64_JUMP26, label_index, 0);

        tcg_out_goto_noaddr(s);

    } else {

        tcg_out_goto(s, l->u.value_ptr);

    }

}
