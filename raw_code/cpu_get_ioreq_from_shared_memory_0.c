static ioreq_t *cpu_get_ioreq_from_shared_memory(XenIOState *state, int vcpu)

{

    ioreq_t *req = xen_vcpu_ioreq(state->shared_page, vcpu);



    if (req->state != STATE_IOREQ_READY) {

        DPRINTF("I/O request not ready: "

                "%x, ptr: %x, port: %"PRIx64", "

                "data: %"PRIx64", count: %" FMT_ioreq_size ", size: %" FMT_ioreq_size "\n",

                req->state, req->data_is_ptr, req->addr,

                req->data, req->count, req->size);

        return NULL;

    }



    xen_rmb(); /* see IOREQ_READY /then/ read contents of ioreq */



    req->state = STATE_IOREQ_INPROCESS;

    return req;

}
