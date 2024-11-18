int64_t qdict_get_try_int(const QDict *qdict, const char *key,

                          int64_t def_value)

{

    QObject *obj;



    obj = qdict_get(qdict, key);

    if (!obj || qobject_type(obj) != QTYPE_QINT)

        return def_value;



    return qint_get_int(qobject_to_qint(obj));

}
