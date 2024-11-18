static void test_visitor_out_empty(TestOutputVisitorData *data,

                                   const void *unused)

{

    QObject *arg;



    arg = qmp_output_get_qobject(data->qov);

    g_assert(qobject_type(arg) == QTYPE_QNULL);



    qobject_decref(arg);

}