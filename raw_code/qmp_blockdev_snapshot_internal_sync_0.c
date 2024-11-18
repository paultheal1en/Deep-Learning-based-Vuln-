void qmp_blockdev_snapshot_internal_sync(const char *device,

                                         const char *name,

                                         Error **errp)

{

    BlockdevSnapshotInternal snapshot = {

        .device = (char *) device,

        .name = (char *) name

    };

    TransactionAction action = {

        .type = TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_INTERNAL_SYNC,

        .u.blockdev_snapshot_internal_sync = &snapshot,

    };

    blockdev_do_action(&action, errp);

}
