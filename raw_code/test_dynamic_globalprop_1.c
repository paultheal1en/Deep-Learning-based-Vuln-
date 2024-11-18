static void test_dynamic_globalprop(void)

{

    MyType *mt;

    static GlobalProperty props[] = {

        { TYPE_DYNAMIC_PROPS, "prop1", "101" },

        { TYPE_DYNAMIC_PROPS, "prop2", "102" },

        { TYPE_DYNAMIC_PROPS"-bad", "prop3", "103", true },

        {}

    };

    int all_used;



    qdev_prop_register_global_list(props);



    mt = DYNAMIC_TYPE(object_new(TYPE_DYNAMIC_PROPS));

    qdev_init_nofail(DEVICE(mt));



    g_assert_cmpuint(mt->prop1, ==, 101);

    g_assert_cmpuint(mt->prop2, ==, 102);



}