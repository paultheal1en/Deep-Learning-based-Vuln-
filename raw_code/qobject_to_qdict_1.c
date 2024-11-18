QDict *qobject_to_qdict(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QDICT)

        return NULL;



    return container_of(obj, QDict, base);

}
