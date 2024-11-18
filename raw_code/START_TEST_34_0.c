START_TEST(qlist_iter_test)

{

    int i;

    QList *qlist;



    qlist = qlist_new();



    for (i = 0; i < iter_max; i++)

        qlist_append(qlist, qint_from_int(i));



    iter_called = 0;

    qlist_iter(qlist, iter_func, NULL);



    fail_unless(iter_called == iter_max);



    QDECREF(qlist);

}
