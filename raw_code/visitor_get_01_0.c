static QObject *visitor_get(TestOutputVisitorData *data)

{

    visit_complete(data->ov, &data->obj);

    g_assert(data->obj);

    return data->obj;

}
