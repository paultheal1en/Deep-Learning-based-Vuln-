static void external_snapshot_commit(BlkActionState *common)

{

    ExternalSnapshotState *state =

                             DO_UPCAST(ExternalSnapshotState, common, common);



    bdrv_set_aio_context(state->new_bs, state->aio_context);



    /* This removes our old bs and adds the new bs */

    bdrv_append(state->new_bs, state->old_bs);

    /* We don't need (or want) to use the transactional

     * bdrv_reopen_multiple() across all the entries at once, because we

     * don't want to abort all of them if one of them fails the reopen */

    bdrv_reopen(state->old_bs, state->old_bs->open_flags & ~BDRV_O_RDWR,

                NULL);

}
