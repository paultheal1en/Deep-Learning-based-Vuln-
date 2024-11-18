void qmp_blockdev_snapshot_sync(bool has_device, const char *device,

                                bool has_node_name, const char *node_name,

                                const char *snapshot_file,

                                bool has_snapshot_node_name,

                                const char *snapshot_node_name,

                                bool has_format, const char *format,

                                bool has_mode, NewImageMode mode, Error **errp)

{

    BlockdevSnapshotSync snapshot = {

        .has_device = has_device,

        .device = (char *) device,

        .has_node_name = has_node_name,

        .node_name = (char *) node_name,

        .snapshot_file = (char *) snapshot_file,

        .has_snapshot_node_name = has_snapshot_node_name,

        .snapshot_node_name = (char *) snapshot_node_name,

        .has_format = has_format,

        .format = (char *) format,

        .has_mode = has_mode,

        .mode = mode,

    };

    TransactionAction action = {

        .type = TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_SYNC,

        .u.blockdev_snapshot_sync = &snapshot,

    };

    blockdev_do_action(&action, errp);

}
