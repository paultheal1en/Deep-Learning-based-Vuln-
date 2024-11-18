char *object_property_get_str(Object *obj, const char *name,

                              Error **errp)

{

    QObject *ret = object_property_get_qobject(obj, name, errp);

    QString *qstring;

    char *retval;



    if (!ret) {

        return NULL;

    }

    qstring = qobject_to_qstring(ret);

    if (!qstring) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name, "string");

        retval = NULL;

    } else {

        retval = g_strdup(qstring_get_str(qstring));

    }



    QDECREF(qstring);

    return retval;

}
