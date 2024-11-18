static void test_abstract_interfaces(void)

{

    QList *all_types;

    QList *obj_types;

    QListEntry *ae;



    qtest_start(common_args);

    /* qom-list-types implements=interface would return any type

     * that implements _any_ interface (not just interface types),

     * so use a trick to find the interface type names:

     * - list all object types

     * - list all types, and look for items that are not

     *   on the first list

     */

    all_types = qom_list_types(NULL, false);

    obj_types = qom_list_types("object", false);



    QLIST_FOREACH_ENTRY(all_types, ae) {

        QDict *at = qobject_to_qdict(qlist_entry_obj(ae));

        const char *aname = qdict_get_str(at, "name");

        QListEntry *oe;

        const char *found = NULL;



        QLIST_FOREACH_ENTRY(obj_types, oe) {

            QDict *ot = qobject_to_qdict(qlist_entry_obj(oe));

            const char *oname = qdict_get_str(ot, "name");

            if (!strcmp(aname, oname)) {

                found = oname;

                break;

            }

        }



        /* Using g_assert_cmpstr() will give more useful failure

         * messages than g_assert(found) */

        g_assert_cmpstr(aname, ==, found);

    }



    QDECREF(all_types);

    QDECREF(obj_types);

    qtest_end();

}
