static void parse_type_number(Visitor *v, const char *name, double *obj,

                              Error **errp)

{

    StringInputVisitor *siv = to_siv(v);

    char *endp = (char *) siv->string;

    double val;



    errno = 0;

    if (siv->string) {

        val = strtod(siv->string, &endp);

    }

    if (!siv->string || errno || endp == siv->string || *endp) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "number");

        return;

    }



    *obj = val;

}
