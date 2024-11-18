int qcow2_cache_flush(BlockDriverState *bs, Qcow2Cache *c)

{

    BDRVQcow2State *s = bs->opaque;

    int result = 0;

    int ret;

    int i;



    trace_qcow2_cache_flush(qemu_coroutine_self(), c == s->l2_table_cache);



    for (i = 0; i < c->size; i++) {

        ret = qcow2_cache_entry_flush(bs, c, i);

        if (ret < 0 && result != -ENOSPC) {

            result = ret;

        }

    }



    if (result == 0) {

        ret = bdrv_flush(bs->file->bs);

        if (ret < 0) {

            result = ret;

        }

    }



    return result;

}
