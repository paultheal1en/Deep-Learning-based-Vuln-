static Visitor *visitor_input_test_init_internal(TestInputVisitorData *data,

                                                 const char *json_string,

                                                 va_list *ap)

{

    visitor_input_teardown(data, NULL);



    data->obj = qobject_from_jsonv(json_string, ap);

    g_assert(data->obj);



    data->qiv = qobject_input_visitor_new(data->obj);

    g_assert(data->qiv);

    return data->qiv;

}
