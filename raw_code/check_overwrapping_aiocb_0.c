static bool check_overwrapping_aiocb(BDRVSheepdogState *s, SheepdogAIOCB *aiocb)

{

    SheepdogAIOCB *cb;



    QLIST_FOREACH(cb, &s->inflight_aiocb_head, aiocb_siblings) {

        if (AIOCBOverwrapping(aiocb, cb)) {

            return true;

        }

    }



    QLIST_INSERT_HEAD(&s->inflight_aiocb_head, aiocb, aiocb_siblings);

    return false;

}
