static void qdict_do_flatten(QDict *qdict, QDict *target, const char *prefix)

{

    QObject *value;

    const QDictEntry *entry, *next;

    const char *new_key;

    bool delete;



    entry = qdict_first(qdict);



    while (entry != NULL) {



        next = qdict_next(qdict, entry);

        value = qdict_entry_value(entry);

        new_key = NULL;

        delete = false;



        if (prefix) {

            qobject_incref(value);

            new_key = g_strdup_printf("%s.%s", prefix, entry->key);

            qdict_put_obj(target, new_key, value);

            delete = true;

        }



        if (qobject_type(value) == QTYPE_QDICT) {

            qdict_do_flatten(qobject_to_qdict(value), target,

                             new_key ? new_key : entry->key);

            delete = true;

        }



        if (delete) {

            qdict_del(qdict, entry->key);



            /* Restart loop after modifying the iterated QDict */

            entry = qdict_first(qdict);

            continue;

        }



        entry = next;

    }

}
