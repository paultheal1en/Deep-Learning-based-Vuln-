int bdrv_create(BlockDriver *drv, const char* filename,

    QEMUOptionParameter *options)

{

    int ret;



    Coroutine *co;

    CreateCo cco = {

        .drv = drv,

        .filename = g_strdup(filename),

        .options = options,

        .ret = NOT_DONE,

    };



    if (!drv->bdrv_create) {

        return -ENOTSUP;

    }



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_create_co_entry(&cco);

    } else {

        co = qemu_coroutine_create(bdrv_create_co_entry);

        qemu_coroutine_enter(co, &cco);

        while (cco.ret == NOT_DONE) {

            qemu_aio_wait();

        }

    }



    ret = cco.ret;

    g_free(cco.filename);



    return ret;

}
