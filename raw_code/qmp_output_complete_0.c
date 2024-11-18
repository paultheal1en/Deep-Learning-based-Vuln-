static void qmp_output_complete(Visitor *v, void *opaque)

{

    QmpOutputVisitor *qov = to_qov(v);



    /* A visit must have occurred, with each start paired with end.  */

    assert(qov->root && QSLIST_EMPTY(&qov->stack));

    assert(opaque == qov->result);



    qobject_incref(qov->root);

    *qov->result = qov->root;

    qov->result = NULL;

}
