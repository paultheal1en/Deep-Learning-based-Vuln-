void input_type_enum(Visitor *v, int *obj, const char *strings[],

                     const char *kind, const char *name,

                     Error **errp)

{

    int64_t value = 0;

    char *enum_str;



    assert(strings);



    visit_type_str(v, &enum_str, name, errp);

    if (error_is_set(errp)) {

        return;

    }



    while (strings[value] != NULL) {

        if (strcmp(strings[value], enum_str) == 0) {

            break;

        }

        value++;

    }



    if (strings[value] == NULL) {

        error_set(errp, QERR_INVALID_PARAMETER, enum_str);

        g_free(enum_str);

        return;

    }



    g_free(enum_str);

    *obj = value;

}
