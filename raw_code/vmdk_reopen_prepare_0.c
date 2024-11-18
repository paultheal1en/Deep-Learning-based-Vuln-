static int vmdk_reopen_prepare(BDRVReopenState *state,

                               BlockReopenQueue *queue, Error **errp)

{

    BDRVVmdkState *s;

    int ret = -1;

    int i;

    VmdkExtent *e;



    assert(state != NULL);

    assert(state->bs != NULL);



    if (queue == NULL) {

        error_setg(errp, "No reopen queue for VMDK extents");

        goto exit;

    }



    s = state->bs->opaque;



    assert(s != NULL);



    for (i = 0; i < s->num_extents; i++) {

        e = &s->extents[i];

        if (e->file != state->bs->file) {

            bdrv_reopen_queue(queue, e->file, state->flags);

        }

    }

    ret = 0;



exit:

    return ret;

}
