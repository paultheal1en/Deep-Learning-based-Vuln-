static int raw_reopen_prepare(BDRVReopenState *reopen_state,

                              BlockReopenQueue *queue, Error **errp)

{

    assert(reopen_state != NULL);

    assert(reopen_state->bs != NULL);



    reopen_state->opaque = g_new0(BDRVRawState, 1);



    return raw_read_options(

        reopen_state->options,

        reopen_state->bs,

        reopen_state->opaque,

        errp);

}
