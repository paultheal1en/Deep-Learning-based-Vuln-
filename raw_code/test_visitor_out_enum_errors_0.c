static void test_visitor_out_enum_errors(TestOutputVisitorData *data,

                                         const void *unused)

{

    EnumOne i, bad_values[] = { ENUM_ONE__MAX, -1 };

    Error *err;



    for (i = 0; i < ARRAY_SIZE(bad_values) ; i++) {

        err = NULL;

        visit_type_EnumOne(data->ov, "unused", &bad_values[i], &err);

        g_assert(err);

        error_free(err);

        visitor_reset(data);

    }

}
