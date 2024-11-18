bool qdict_get_try_bool(const QDict *qdict, const char *key, bool def_value)

{

    QObject *obj;



    obj = qdict_get(qdict, key);

    if (!obj || qobject_type(obj) != QTYPE_QBOOL)

        return def_value;



    return qbool_get_bool(qobject_to_qbool(obj));

}
