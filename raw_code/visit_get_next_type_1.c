void visit_get_next_type(Visitor *v, int *obj, const int *qtypes,

                         const char *name, Error **errp)

{

    if (v->get_next_type) {

        v->get_next_type(v, obj, qtypes, name, errp);

    }

}
