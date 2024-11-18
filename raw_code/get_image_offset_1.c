static inline int64_t get_image_offset(BlockDriverState *bs, uint64_t offset,

                                       bool write)

{

    BDRVVPCState *s = bs->opaque;

    uint64_t bitmap_offset, block_offset;

    uint32_t pagetable_index, offset_in_block;



    pagetable_index = offset / s->block_size;

    offset_in_block = offset % s->block_size;



    if (pagetable_index >= s->max_table_entries || s->pagetable[pagetable_index] == 0xffffffff)

        return -1; /* not allocated */



    bitmap_offset = 512 * (uint64_t) s->pagetable[pagetable_index];

    block_offset = bitmap_offset + s->bitmap_size + offset_in_block;



    /* We must ensure that we don't write to any sectors which are marked as

       unused in the bitmap. We get away with setting all bits in the block

       bitmap each time we write to a new block. This might cause Virtual PC to

       miss sparse read optimization, but it's not a problem in terms of

       correctness. */

    if (write && (s->last_bitmap_offset != bitmap_offset)) {

        uint8_t bitmap[s->bitmap_size];



        s->last_bitmap_offset = bitmap_offset;

        memset(bitmap, 0xff, s->bitmap_size);

        bdrv_pwrite_sync(bs->file, bitmap_offset, bitmap, s->bitmap_size);

    }



    return block_offset;

}
