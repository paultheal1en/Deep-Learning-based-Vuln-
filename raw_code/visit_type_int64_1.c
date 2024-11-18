void visit_type_int64(Visitor *v, int64_t *obj, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        if (v->type_int64) {

            v->type_int64(v, obj, name, errp);

        } else {

            v->type_int(v, obj, name, errp);

        }

    }

}
