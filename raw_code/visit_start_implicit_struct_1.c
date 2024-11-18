void visit_start_implicit_struct(Visitor *v, void **obj, size_t size,

                                 Error **errp)

{

    if (!error_is_set(errp) && v->start_implicit_struct) {

        v->start_implicit_struct(v, obj, size, errp);

    }

}
