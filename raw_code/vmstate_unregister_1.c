void vmstate_unregister(DeviceState *dev, const VMStateDescription *vmsd,

                        void *opaque)

{

    SaveStateEntry *se, *new_se;



    QTAILQ_FOREACH_SAFE(se, &savevm_handlers, entry, new_se) {

        if (se->vmsd == vmsd && se->opaque == opaque) {

            QTAILQ_REMOVE(&savevm_handlers, se, entry);




            qemu_free(se);


