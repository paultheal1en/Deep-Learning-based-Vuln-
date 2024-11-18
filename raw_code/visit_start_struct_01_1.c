void visit_start_struct(Visitor *v, const char *name, void **obj,

                        size_t size, Error **errp)

{

    Error *err = NULL;



    if (obj) {

        assert(size);

        assert(v->type != VISITOR_OUTPUT || *obj);

    }

    v->start_struct(v, name, obj, size, &err);

    if (obj && v->type == VISITOR_INPUT) {

        assert(!err != !*obj);

    }

    error_propagate(errp, err);

}
