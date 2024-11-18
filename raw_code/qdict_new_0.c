QDict *qdict_new(void)

{

    QDict *qdict;



    qdict = g_malloc0(sizeof(*qdict));

    QOBJECT_INIT(qdict, &qdict_type);



    return qdict;

}
