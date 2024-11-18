bool bdrv_op_is_blocked(BlockDriverState *bs, BlockOpType op, Error **errp)

{

    BdrvOpBlocker *blocker;

    assert((int) op >= 0 && op < BLOCK_OP_TYPE_MAX);

    if (!QLIST_EMPTY(&bs->op_blockers[op])) {

        blocker = QLIST_FIRST(&bs->op_blockers[op]);

        if (errp) {

            *errp = error_copy(blocker->reason);

            error_prepend(errp, "Node '%s' is busy: ",

                          bdrv_get_device_or_node_name(bs));

        }

        return true;

    }

    return false;

}
