static void qed_read_l2_table_cb(void *opaque, int ret)

{

    QEDReadL2TableCB *read_l2_table_cb = opaque;

    QEDRequest *request = read_l2_table_cb->request;

    BDRVQEDState *s = read_l2_table_cb->s;

    CachedL2Table *l2_table = request->l2_table;



    if (ret) {

        /* can't trust loaded L2 table anymore */

        qed_unref_l2_cache_entry(l2_table);

        request->l2_table = NULL;

    } else {

        l2_table->offset = read_l2_table_cb->l2_offset;



        qed_commit_l2_cache_entry(&s->l2_cache, l2_table);



        /* This is guaranteed to succeed because we just committed the entry

         * to the cache.

         */

        request->l2_table = qed_find_l2_cache_entry(&s->l2_cache,

                                                    l2_table->offset);

        assert(request->l2_table != NULL);

    }



    gencb_complete(&read_l2_table_cb->gencb, ret);

}
