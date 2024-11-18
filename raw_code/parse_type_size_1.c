static void parse_type_size(Visitor *v, const char *name, uint64_t *obj,

                            Error **errp)

{

    StringInputVisitor *siv = to_siv(v);

    Error *err = NULL;

    uint64_t val;



    if (siv->string) {

        parse_option_size(name, siv->string, &val, &err);

    } else {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "size");

        return;

    }

    if (err) {

        error_propagate(errp, err);

        return;

    }



    *obj = val;

}
