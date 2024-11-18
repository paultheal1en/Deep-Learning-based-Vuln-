void visit_end_implicit_struct(Visitor *v, Error **errp)

{

    assert(!error_is_set(errp));

    if (v->end_implicit_struct) {

        v->end_implicit_struct(v, errp);

    }

}
