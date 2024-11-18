void vmstate_unregister(DeviceState *dev, const VMStateDescription *vmsd,

                        void *opaque)

{

    SaveStateEntry *se, *new_se;



    QTAILQ_FOREACH_SAFE(se, &savevm_state.handlers, entry, new_se) {

        if (se->vmsd == vmsd && se->opaque == opaque) {

            QTAILQ_REMOVE(&savevm_state.handlers, se, entry);

            if (se->compat) {

                g_free(se->compat);

            }

            g_free(se);

        }

    }

}
