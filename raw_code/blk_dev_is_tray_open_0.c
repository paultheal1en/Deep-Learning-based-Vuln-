bool blk_dev_is_tray_open(BlockBackend *blk)

{

    if (blk->dev_ops && blk->dev_ops->is_tray_open) {

        return blk->dev_ops->is_tray_open(blk->dev_opaque);

    }

    return false;

}
