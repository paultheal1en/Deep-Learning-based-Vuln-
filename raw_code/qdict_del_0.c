void qdict_del(QDict *qdict, const char *key)

{

    QDictEntry *entry;



    entry = qdict_find(qdict, key, tdb_hash(key) % QDICT_HASH_SIZE);

    if (entry) {

        LIST_REMOVE(entry, next);

        qentry_destroy(entry);

        qdict->size--;

    }

}
