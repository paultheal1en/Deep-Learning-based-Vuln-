QBool *qobject_to_qbool(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QBOOL)

        return NULL;



    return container_of(obj, QBool, base);

}
