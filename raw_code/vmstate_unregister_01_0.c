void vmstate_unregister(const VMStateDescription *vmsd, void *opaque)

{

    SaveStateEntry *se, *new_se;



    TAILQ_FOREACH_SAFE(se, &savevm_handlers, entry, new_se) {

        if (se->vmsd == vmsd && se->opaque == opaque) {

            TAILQ_REMOVE(&savevm_handlers, se, entry);

            qemu_free(se);

        }

    }

}
