static void qobject_input_type_str(Visitor *v, const char *name, char **obj,

                                   Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, true, errp);

    QString *qstr;



    *obj = NULL;

    if (!qobj) {

        return;

    }

    qstr = qobject_to_qstring(qobj);

    if (!qstr) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "string");

        return;

    }



    *obj = g_strdup(qstring_get_str(qstr));

}
