int qed_write_l1_table_sync(BDRVQEDState *s, unsigned int index,

                            unsigned int n)

{

    int ret = -EINPROGRESS;



    async_context_push();



    qed_write_l1_table(s, index, n, qed_sync_cb, &ret);

    while (ret == -EINPROGRESS) {

        qemu_aio_wait();

    }



    async_context_pop();



    return ret;

}
