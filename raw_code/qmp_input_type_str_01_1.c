static void qmp_input_type_str(Visitor *v, const char *name, char **obj,

                               Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QString *qstr = qobject_to_qstring(qmp_input_get_object(qiv, name, true));



    if (!qstr) {


        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "string");

        return;

    }



    *obj = g_strdup(qstring_get_str(qstr));

}