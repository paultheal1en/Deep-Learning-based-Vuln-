int qed_write_l2_table_sync(BDRVQEDState *s, QEDRequest *request,

                            unsigned int index, unsigned int n, bool flush)

{

    int ret = -EINPROGRESS;



    qed_write_l2_table(s, request, index, n, flush, qed_sync_cb, &ret);

    while (ret == -EINPROGRESS) {

        aio_poll(bdrv_get_aio_context(s->bs), true);

    }



    return ret;

}
