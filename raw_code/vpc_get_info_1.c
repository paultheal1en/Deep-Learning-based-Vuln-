static int vpc_get_info(BlockDriverState *bs, BlockDriverInfo *bdi)

{

    BDRVVPCState *s = (BDRVVPCState *)bs->opaque;

    VHDFooter *footer = (VHDFooter *) s->footer_buf;



    if (cpu_to_be32(footer->type) != VHD_FIXED) {

        bdi->cluster_size = s->block_size;

    }



    bdi->unallocated_blocks_are_zero = true;

    return 0;

}
