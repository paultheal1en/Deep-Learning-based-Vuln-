static void test_struct(gconstpointer opaque)

{

    TestArgs *args = (TestArgs *) opaque;

    const SerializeOps *ops = args->ops;

    TestStruct *ts = struct_create();

    TestStruct *ts_copy = NULL;

    Error *err = NULL;

    void *serialize_data;



    ops->serialize(ts, &serialize_data, visit_struct, &err);

    ops->deserialize((void **)&ts_copy, serialize_data, visit_struct, &err); 



    g_assert(err == NULL);

    struct_compare(ts, ts_copy);



    struct_cleanup(ts);

    struct_cleanup(ts_copy);



    ops->cleanup(serialize_data);

    g_free(args);

}
