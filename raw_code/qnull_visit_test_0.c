static void qnull_visit_test(void)

{

    QObject *obj;

    QmpOutputVisitor *qov;

    Visitor *v;



    /*

     * Most tests of interactions between QObject and visitors are in

     * test-qmp-*-visitor; but these tests live here because they

     * depend on layering violations to check qnull_ refcnt.

     */



    g_assert(qnull_.refcnt == 1);

    obj = qnull();

    v = qmp_input_visitor_new(obj, true);

    qobject_decref(obj);

    visit_type_null(v, NULL, &error_abort);

    visit_free(v);



    qov = qmp_output_visitor_new();

    visit_type_null(qmp_output_get_visitor(qov), NULL, &error_abort);

    obj = qmp_output_get_qobject(qov);

    g_assert(obj == &qnull_);

    qobject_decref(obj);

    qmp_output_visitor_cleanup(qov);



    g_assert(qnull_.refcnt == 1);

}
