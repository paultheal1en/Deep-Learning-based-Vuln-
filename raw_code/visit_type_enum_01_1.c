void visit_type_enum(Visitor *v, int *obj, const char *strings[],

                     const char *kind, const char *name, Error **errp)

{

    if (!error_is_set(errp)) {

        v->type_enum(v, obj, strings, kind, name, errp);

    }

}
