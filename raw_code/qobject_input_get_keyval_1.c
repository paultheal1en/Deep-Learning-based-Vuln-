static const char *qobject_input_get_keyval(QObjectInputVisitor *qiv,

                                            const char *name,

                                            Error **errp)

{

    QObject *qobj;

    QString *qstr;



    qobj = qobject_input_get_object(qiv, name, true, errp);

    if (!qobj) {

        return NULL;

    }



    qstr = qobject_to_qstring(qobj);

    if (!qstr) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE,

                   full_name(qiv, name), "string");

        return NULL;

    }



    return qstring_get_str(qstr);

}
