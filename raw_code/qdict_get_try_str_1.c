const char *qdict_get_try_str(const QDict *qdict, const char *key)

{

    QObject *obj;



    obj = qdict_get(qdict, key);

    if (!obj || qobject_type(obj) != QTYPE_QSTRING)

        return NULL;



    return qstring_get_str(qobject_to_qstring(obj));

}
