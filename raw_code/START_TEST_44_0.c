START_TEST(qlist_new_test)

{

    QList *qlist;



    qlist = qlist_new();

    fail_unless(qlist != NULL);

    fail_unless(qlist->base.refcnt == 1);

    fail_unless(qobject_type(QOBJECT(qlist)) == QTYPE_QLIST);



    // destroy doesn't exist yet

    g_free(qlist);

}
