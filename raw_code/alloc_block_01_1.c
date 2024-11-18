static int64_t alloc_block(BlockDriverState* bs, int64_t sector_num)

{

    BDRVVPCState *s = bs->opaque;

    int64_t bat_offset;

    uint32_t index, bat_value;

    int ret;

    uint8_t bitmap[s->bitmap_size];



    // Check if sector_num is valid

    if ((sector_num < 0) || (sector_num > bs->total_sectors))

        return -1;



    // Write entry into in-memory BAT

    index = (sector_num * 512) / s->block_size;

    if (s->pagetable[index] != 0xFFFFFFFF)

        return -1;



    s->pagetable[index] = s->free_data_block_offset / 512;



    // Initialize the block's bitmap

    memset(bitmap, 0xff, s->bitmap_size);

    bdrv_pwrite(bs->file, s->free_data_block_offset, bitmap, s->bitmap_size);



    // Write new footer (the old one will be overwritten)

    s->free_data_block_offset += s->block_size + s->bitmap_size;

    ret = rewrite_footer(bs);

    if (ret < 0)

        goto fail;



    // Write BAT entry to disk

    bat_offset = s->bat_offset + (4 * index);

    bat_value = be32_to_cpu(s->pagetable[index]);

    ret = bdrv_pwrite(bs->file, bat_offset, &bat_value, 4);

    if (ret < 0)

        goto fail;



    return get_sector_offset(bs, sector_num, 0);



fail:

    s->free_data_block_offset -= (s->block_size + s->bitmap_size);

    return -1;

}
