static void add_machine_test_cases(void)

{

    const char *arch = qtest_get_arch();

    QDict *response, *minfo;

    QList *list;

    const QListEntry *p;

    QObject *qobj;

    QString *qstr;

    const char *mname, *path;



    qtest_start("-machine none");

    response = qmp("{ 'execute': 'query-machines' }");

    g_assert(response);

    list = qdict_get_qlist(response, "return");

    g_assert(list);



    for (p = qlist_first(list); p; p = qlist_next(p)) {

        minfo = qobject_to_qdict(qlist_entry_obj(p));

        g_assert(minfo);

        qobj = qdict_get(minfo, "name");

        g_assert(qobj);

        qstr = qobject_to_qstring(qobj);

        g_assert(qstr);

        mname = qstring_get_str(qstr);

        if (!is_blacklisted(arch, mname)) {

            path = g_strdup_printf("qom/%s", mname);

            qtest_add_data_func(path, g_strdup(mname), test_machine);

        }

    }



    qtest_end();

    QDECREF(response);

}
