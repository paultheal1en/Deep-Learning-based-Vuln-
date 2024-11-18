static void qmp_output_push_obj(QmpOutputVisitor *qov, QObject *value,

                                void *qapi)

{

    QStackEntry *e = g_malloc0(sizeof(*e));



    assert(qov->root);

    assert(value);

    e->value = value;

    e->qapi = qapi;

    QSLIST_INSERT_HEAD(&qov->stack, e, node);

}
