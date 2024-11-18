static const QObject *qmp_input_get_object(QmpInputVisitor *qiv,

                                           const char *name)

{

    const QObject *qobj;



    if (qiv->nb_stack == 0) {

        qobj = qiv->obj;

    } else {

        qobj = qiv->stack[qiv->nb_stack - 1].obj;

    }



    if (name && qobject_type(qobj) == QTYPE_QDICT) {

        return qdict_get(qobject_to_qdict(qobj), name);

    } else if (qiv->nb_stack > 0 && qobject_type(qobj) == QTYPE_QLIST) {

        return qlist_entry_obj(qiv->stack[qiv->nb_stack - 1].entry);

    }



    return qobj;

}
