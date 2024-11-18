static void visitor_output_setup_internal(TestOutputVisitorData *data,

                                          bool human)

{

    data->human = human;

    data->sov = string_output_visitor_new(human);

    g_assert(data->sov);

    data->ov = string_output_get_visitor(data->sov);

    g_assert(data->ov);

}
