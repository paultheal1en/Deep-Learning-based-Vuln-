static void qmp_output_add_obj(QmpOutputVisitor *qov, const char *name,

                               QObject *value)

{

    QStackEntry *e = QTAILQ_FIRST(&qov->stack);

    QObject *cur = e ? e->value : NULL;



    if (!cur) {

        /* FIXME we should require the user to reset the visitor, rather

         * than throwing away the previous root */

        qobject_decref(qov->root);

        qov->root = value;

    } else {

        switch (qobject_type(cur)) {

        case QTYPE_QDICT:

            assert(name);

            qdict_put_obj(qobject_to_qdict(cur), name, value);

            break;

        case QTYPE_QLIST:

            qlist_append_obj(qobject_to_qlist(cur), value);

            break;

        default:

            g_assert_not_reached();

        }

    }

}
