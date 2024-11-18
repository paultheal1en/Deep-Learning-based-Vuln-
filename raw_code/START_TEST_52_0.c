START_TEST(qlist_append_test)

{

    QInt *qi;

    QList *qlist;

    QListEntry *entry;



    qi = qint_from_int(42);



    qlist = qlist_new();

    qlist_append(qlist, qi);



    entry = QTAILQ_FIRST(&qlist->head);

    fail_unless(entry != NULL);

    fail_unless(entry->value == QOBJECT(qi));



    // destroy doesn't exist yet

    QDECREF(qi);

    g_free(entry);

    g_free(qlist);

}
