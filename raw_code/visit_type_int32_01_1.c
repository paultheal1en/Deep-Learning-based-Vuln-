static void visit_type_int32(Visitor *v, int *value, const char *name, Error **errp)

{

    int64_t val = *value;

    visit_type_int(v, &val, name, errp);

}
