static void external_snapshot_abort(BlkActionState *common)

{

    ExternalSnapshotState *state =

                             DO_UPCAST(ExternalSnapshotState, common, common);

    if (state->new_bs) {

        if (state->new_bs->backing) {

            bdrv_replace_in_backing_chain(state->new_bs, state->old_bs);

        }

    }

}
