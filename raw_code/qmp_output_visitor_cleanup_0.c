void qmp_output_visitor_cleanup(QmpOutputVisitor *v)

{

    QStackEntry *e, *tmp;



    QTAILQ_FOREACH_SAFE(e, &v->stack, node, tmp) {

        QTAILQ_REMOVE(&v->stack, e, node);

        g_free(e);

    }



    qobject_decref(v->root);

    g_free(v);

}
