static void qdict_teardown(void)

{

    QDECREF(tests_dict);

    tests_dict = NULL;

}
