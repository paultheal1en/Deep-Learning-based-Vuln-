static void qmp_input_pop(Visitor *v, void **obj)

{

    QmpInputVisitor *qiv = to_qiv(v);

    StackObject *tos = QSLIST_FIRST(&qiv->stack);



    assert(tos && tos->qapi == obj);

    QSLIST_REMOVE_HEAD(&qiv->stack, node);

    qmp_input_stack_object_free(tos);

}
