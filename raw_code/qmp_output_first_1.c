static QObject *qmp_output_first(QmpOutputVisitor *qov)

{

    QStackEntry *e = QTAILQ_LAST(&qov->stack, QStack);



    /* FIXME - find a better way to deal with NULL values */

    if (!e) {

        return NULL;

    }



    return e->value;

}
