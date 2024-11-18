static void rndis_clear_responsequeue(USBNetState *s)

{

    struct rndis_response *r;



    while ((r = s->rndis_resp.tqh_first)) {

        TAILQ_REMOVE(&s->rndis_resp, r, entries);

        qemu_free(r);

    }

}
