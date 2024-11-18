static void tcg_out_label(TCGContext *s, int label_index, tcg_insn_unit *ptr)

{

    TCGLabel *l = &s->labels[label_index];

    intptr_t value = (intptr_t)ptr;

    TCGRelocation *r;



    assert(!l->has_value);



    for (r = l->u.first_reloc; r != NULL; r = r->next) {

        patch_reloc(r->ptr, r->type, value, r->addend);

    }



    l->has_value = 1;

    l->u.value_ptr = ptr;

}
