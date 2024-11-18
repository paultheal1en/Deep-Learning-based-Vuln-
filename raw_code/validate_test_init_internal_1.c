static Visitor *validate_test_init_internal(TestInputVisitorData *data,
                                            const char *json_string,
                                            va_list *ap)
{
    Visitor *v;
    data->obj = qobject_from_jsonv(json_string, ap);
    g_assert(data->obj);
    data->qiv = qmp_input_visitor_new_strict(data->obj);
    g_assert(data->qiv);
    v = qmp_input_get_visitor(data->qiv);
    g_assert(v);
    return v;
}