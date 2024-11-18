static void do_test_validate_qmp_introspect(TestInputVisitorData *data,

                                            const char *schema_json)

{

    SchemaInfoList *schema = NULL;

    Visitor *v;



    v = validate_test_init_raw(data, schema_json);



    visit_type_SchemaInfoList(v, NULL, &schema, &error_abort);

    g_assert(schema);



    qapi_free_SchemaInfoList(schema);

}
