static void test_primitives(gconstpointer opaque)

{

    TestArgs *args = (TestArgs *) opaque;

    const SerializeOps *ops = args->ops;

    PrimitiveType *pt = args->test_data;

    PrimitiveType *pt_copy = g_malloc0(sizeof(*pt_copy));

    Error *err = NULL;

    void *serialize_data;

    char *double1, *double2;



    pt_copy->type = pt->type;

    ops->serialize(pt, &serialize_data, visit_primitive_type, &err);

    ops->deserialize((void **)&pt_copy, serialize_data, visit_primitive_type, &err);



    g_assert(err == NULL);

    g_assert(pt_copy != NULL);

    if (pt->type == PTYPE_STRING) {

        g_assert_cmpstr(pt->value.string, ==, pt_copy->value.string);

        g_free((char *)pt_copy->value.string);

    } else if (pt->type == PTYPE_NUMBER) {

        /* we serialize with %f for our reference visitors, so rather than fuzzy

         * floating math to test "equality", just compare the formatted values

         */

        double1 = g_malloc0(calc_float_string_storage(pt->value.number));

        double2 = g_malloc0(calc_float_string_storage(pt_copy->value.number));

        g_assert_cmpstr(double1, ==, double2);

        g_free(double1);

        g_free(double2);

    } else if (pt->type == PTYPE_BOOLEAN) {

        g_assert_cmpint(!!pt->value.max, ==, !!pt->value.max);

    } else {

        g_assert_cmpint(pt->value.max, ==, pt_copy->value.max);

    }



    ops->cleanup(serialize_data);

    g_free(args);

    g_free(pt_copy);

}
