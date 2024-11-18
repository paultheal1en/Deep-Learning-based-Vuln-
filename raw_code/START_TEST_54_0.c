START_TEST(qdict_del_test)

{

    const char *key = "key test";



    qdict_put(tests_dict, key, qstring_from_str("foo"));

    fail_unless(qdict_size(tests_dict) == 1);



    qdict_del(tests_dict, key);



    fail_unless(qdict_size(tests_dict) == 0);

    fail_unless(qdict_haskey(tests_dict, key) == 0);

}
