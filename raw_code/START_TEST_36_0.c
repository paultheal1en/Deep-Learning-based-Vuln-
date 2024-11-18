START_TEST(qdict_stress_test)

{

    size_t lines;

    char key[128];

    FILE *test_file;

    QDict *qdict;

    QString *value;

    const char *test_file_path = "qdict-test-data.txt";



    test_file = fopen(test_file_path, "r");

    fail_unless(test_file != NULL);



    // Create the dict

    qdict = qdict_new();

    fail_unless(qdict != NULL);



    // Add everything from the test file

    for (lines = 0;; lines++) {

        value = read_line(test_file, key);

        if (!value)

            break;



        qdict_put(qdict, key, value);

    }

    fail_unless(qdict_size(qdict) == lines);



    // Check if everything is really in there

    reset_file(test_file);

    for (;;) {

        const char *str1, *str2;



        value = read_line(test_file, key);

        if (!value)

            break;



        str1 = qstring_get_str(value);



        str2 = qdict_get_str(qdict, key);

        fail_unless(str2 != NULL);



        fail_unless(strcmp(str1, str2) == 0);



        QDECREF(value);

    }



    // Delete everything

    reset_file(test_file);

    for (;;) {

        value = read_line(test_file, key);

        if (!value)

            break;



        qdict_del(qdict, key);

        QDECREF(value);



        fail_unless(qdict_haskey(qdict, key) == 0);

    }

    fclose(test_file);



    fail_unless(qdict_size(qdict) == 0);

    QDECREF(qdict);

}
