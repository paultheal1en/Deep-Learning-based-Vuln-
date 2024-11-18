static QObject *qmp_output_pop(QmpOutputVisitor *qov, void *qapi)

{

    QStackEntry *e = QSLIST_FIRST(&qov->stack);

    QObject *value;



    assert(e);

    assert(e->qapi == qapi);

    QSLIST_REMOVE_HEAD(&qov->stack, node);

    value = e->value;

    assert(value);

    g_free(e);

    return value;

}
