static void raw_reopen_commit(BDRVReopenState *state)

{

    BDRVRawState *new_s = state->opaque;

    BDRVRawState *s = state->bs->opaque;



    memcpy(s, new_s, sizeof(BDRVRawState));



    g_free(state->opaque);

    state->opaque = NULL;

}
