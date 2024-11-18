START_TEST(qdict_haskey_not_test)

{

    fail_unless(qdict_haskey(tests_dict, "test") == 0);

}
