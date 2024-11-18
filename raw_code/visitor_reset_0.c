static void visitor_reset(TestOutputVisitorData *data)

{

    visitor_output_teardown(data, NULL);

    visitor_output_setup(data, NULL);

}
