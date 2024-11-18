static void parse_type_str(Visitor *v, const char *name, char **obj,

                           Error **errp)

{

    StringInputVisitor *siv = to_siv(v);

    if (siv->string) {

        *obj = g_strdup(siv->string);

    } else {

        *obj = NULL;

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "string");

    }

}
