static void glfs_clear_preopened(glfs_t *fs)

{

    ListElement *entry = NULL;



    if (fs == NULL) {

        return;

    }



    QLIST_FOREACH(entry, &glfs_list, list) {

        if (entry->saved.fs == fs) {

            if (--entry->saved.ref) {

                return;

            }



            QLIST_REMOVE(entry, list);



            glfs_fini(entry->saved.fs);

            g_free(entry->saved.volume);

            g_free(entry);

        }

    }

}
