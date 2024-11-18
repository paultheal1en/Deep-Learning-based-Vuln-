START_TEST(qdict_iterapi_test)

{

    int count;

    const QDictEntry *ent;



    fail_unless(qdict_first(tests_dict) == NULL);



    qdict_put(tests_dict, "key1", qint_from_int(1));

    qdict_put(tests_dict, "key2", qint_from_int(2));

    qdict_put(tests_dict, "key3", qint_from_int(3));



    count = 0;

    for (ent = qdict_first(tests_dict); ent; ent = qdict_next(tests_dict, ent)){

        fail_unless(qdict_haskey(tests_dict, qdict_entry_key(ent)) == 1);

        count++;

    }



    fail_unless(count == qdict_size(tests_dict));



    /* Do it again to test restarting */

    count = 0;

    for (ent = qdict_first(tests_dict); ent; ent = qdict_next(tests_dict, ent)){

        fail_unless(qdict_haskey(tests_dict, qdict_entry_key(ent)) == 1);

        count++;

    }



    fail_unless(count == qdict_size(tests_dict));

}
