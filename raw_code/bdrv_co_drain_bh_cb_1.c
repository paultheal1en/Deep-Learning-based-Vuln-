static void bdrv_co_drain_bh_cb(void *opaque)

{

    BdrvCoDrainData *data = opaque;

    Coroutine *co = data->co;



    qemu_bh_delete(data->bh);

    bdrv_drain_poll(data->bs);

    data->done = true;

    qemu_coroutine_enter(co, NULL);

}
