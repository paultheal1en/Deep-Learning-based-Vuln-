void qmp_blockdev_snapshot(const char *node, const char *overlay,

                           Error **errp)

{

    BlockdevSnapshot snapshot_data = {

        .node = (char *) node,

        .overlay = (char *) overlay

    };

    TransactionAction action = {

        .type = TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT,

        .u.blockdev_snapshot = &snapshot_data,

    };

    blockdev_do_action(&action, errp);

}
