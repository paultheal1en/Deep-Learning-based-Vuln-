void unregister_savevm(const char *idstr, void *opaque)

{

    SaveStateEntry *se, *new_se;



    TAILQ_FOREACH_SAFE(se, &savevm_handlers, entry, new_se) {

        if (strcmp(se->idstr, idstr) == 0 && se->opaque == opaque) {

            TAILQ_REMOVE(&savevm_handlers, se, entry);

            qemu_free(se);

        }

    }

}
