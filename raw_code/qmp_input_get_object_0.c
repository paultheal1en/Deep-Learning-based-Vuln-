static QObject *qmp_input_get_object(QmpInputVisitor *qiv,

                                     const char *name,

                                     bool consume)

{

    StackObject *tos;

    QObject *qobj;

    QObject *ret;



    if (!qiv->nb_stack) {

        /* Starting at root, name is ignored. */

        return qiv->root;

    }



    /* We are in a container; find the next element. */

    tos = &qiv->stack[qiv->nb_stack - 1];

    qobj = tos->obj;

    assert(qobj);



    if (qobject_type(qobj) == QTYPE_QDICT) {

        assert(name);

        ret = qdict_get(qobject_to_qdict(qobj), name);

        if (tos->h && consume && ret) {

            bool removed = g_hash_table_remove(tos->h, name);

            assert(removed);

        }

    } else {

        assert(qobject_type(qobj) == QTYPE_QLIST);

        assert(!name);

        ret = qlist_entry_obj(tos->entry);

        if (consume) {

            tos->entry = qlist_next(tos->entry);

        }

    }



    return ret;

}
