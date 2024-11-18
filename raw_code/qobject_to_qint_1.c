QInt *qobject_to_qint(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QINT)

        return NULL;



    return container_of(obj, QInt, base);

}
