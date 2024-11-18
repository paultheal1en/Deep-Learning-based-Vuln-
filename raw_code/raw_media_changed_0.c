static int raw_media_changed(BlockDriverState *bs)

{

    return bdrv_media_changed(bs->file->bs);

}
