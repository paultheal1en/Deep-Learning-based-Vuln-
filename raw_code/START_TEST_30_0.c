START_TEST(qdict_new_test)

{

    QDict *qdict;



    qdict = qdict_new();

    fail_unless(qdict != NULL);

    fail_unless(qdict_size(qdict) == 0);

    fail_unless(qdict->base.refcnt == 1);

    fail_unless(qobject_type(QOBJECT(qdict)) == QTYPE_QDICT);



    // destroy doesn't exit yet

    free(qdict);

}
