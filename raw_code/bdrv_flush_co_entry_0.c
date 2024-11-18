static void coroutine_fn bdrv_flush_co_entry(void *opaque)

{

    RwCo *rwco = opaque;



    rwco->ret = bdrv_co_flush(rwco->bs);

}
