static void visitor_input_teardown(TestInputVisitorData *data,

                                   const void *unused)

{

    qobject_decref(data->obj);

    data->obj = NULL;



    if (data->qiv) {

        visit_free(data->qiv);

        data->qiv = NULL;

    }

}
