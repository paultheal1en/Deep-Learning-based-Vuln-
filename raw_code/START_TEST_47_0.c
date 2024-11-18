START_TEST(qlist_destroy_test)

{

    int i;

    QList *qlist;



    qlist = qlist_new();



    for (i = 0; i < 42; i++)

        qlist_append(qlist, qint_from_int(i));



    QDECREF(qlist);

}
