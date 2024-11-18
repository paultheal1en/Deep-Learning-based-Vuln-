static Visitor *visitor_input_test_init_raw(TestInputVisitorData *data,

                                            const char *json_string)

{

    return visitor_input_test_init_internal(data, json_string, NULL);

}
