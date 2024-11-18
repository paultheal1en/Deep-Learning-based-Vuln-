QError *qerror_new(void)

{

    QError *qerr;



    qerr = g_malloc0(sizeof(*qerr));

    QOBJECT_INIT(qerr, &qerror_type);



    return qerr;

}
