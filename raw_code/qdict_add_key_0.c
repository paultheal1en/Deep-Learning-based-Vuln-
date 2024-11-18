static void qdict_add_key(const char *key, QObject *obj, void *opaque)

{

    GHashTable *h = opaque;

    g_hash_table_insert(h, (gpointer) key, NULL);

}
