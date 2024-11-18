static void test_properties(const char *path)

{

    char *child_path;

    QDict *response, *tuple;

    QList *list;

    QListEntry *entry;



    g_test_message("Obtaining properties of %s", path);

    response = qmp("{ 'execute': 'qom-list',"

                   "  'arguments': { 'path': '%s' } }", path);

    g_assert(response);



    g_assert(qdict_haskey(response, "return"));

    list = qobject_to_qlist(qdict_get(response, "return"));

    QLIST_FOREACH_ENTRY(list, entry) {

        tuple = qobject_to_qdict(qlist_entry_obj(entry));

        if (strstart(qdict_get_str(tuple, "type"), "child<", NULL)) {

            child_path = g_strdup_printf("%s/%s",

                                         path, qdict_get_str(tuple, "name"));

            test_properties(child_path);

            g_free(child_path);

        } else {

            const char *prop = qdict_get_str(tuple, "name");

            g_test_message("Testing property %s.%s", path, prop);

            response = qmp("{ 'execute': 'qom-get',"

                           "  'arguments': { 'path': '%s',"

                           "                 'property': '%s' } }",

                           path, prop);

            /* qom-get may fail but should not, e.g., segfault. */

            g_assert(response);

        }

    }

}
