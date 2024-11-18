static void qmp_input_push(QmpInputVisitor *qiv, const QObject *obj, Error **errp)

{

    qiv->stack[qiv->nb_stack].obj = obj;

    if (qobject_type(obj) == QTYPE_QLIST) {

        qiv->stack[qiv->nb_stack].entry = qlist_first(qobject_to_qlist(obj));

    }

    qiv->nb_stack++;



    if (qiv->nb_stack >= QIV_STACK_SIZE) {

        error_set(errp, QERR_BUFFER_OVERRUN);

        return;

    }

}
