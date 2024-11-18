void visit_type_bool(Visitor *v, bool *obj, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->type_bool(v, obj, name, errp);

    }

}
