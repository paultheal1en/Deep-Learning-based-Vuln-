static void netfilter_finalize(Object *obj)

{

    NetFilterState *nf = NETFILTER(obj);

    NetFilterClass *nfc = NETFILTER_GET_CLASS(obj);



    if (nfc->cleanup) {

        nfc->cleanup(nf);

    }



    if (nf->netdev && !QTAILQ_EMPTY(&nf->netdev->filters) &&

        nf->next.tqe_prev) {

        QTAILQ_REMOVE(&nf->netdev->filters, nf, next);

    }

    g_free(nf->netdev_id);

}
