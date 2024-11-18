START_TEST(qdict_haskey_test)

{

    const char *key = "test";



    qdict_put(tests_dict, key, qint_from_int(0));

    fail_unless(qdict_haskey(tests_dict, key) == 1);

}
