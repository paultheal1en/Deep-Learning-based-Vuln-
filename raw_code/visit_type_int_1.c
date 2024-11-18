void visit_type_int(Visitor *v, int64_t *obj, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->type_int(v, obj, name, errp);

    }

}
