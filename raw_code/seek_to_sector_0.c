static inline int seek_to_sector(BlockDriverState *bs, int64_t sector_num)

{

    BDRVBochsState *s = bs->opaque;

    int64_t offset = sector_num * 512;

    int64_t extent_index, extent_offset, bitmap_offset, block_offset;

    char bitmap_entry;



    // seek to sector

    extent_index = offset / s->extent_size;

    extent_offset = (offset % s->extent_size) / 512;



    if (s->catalog_bitmap[extent_index] == 0xffffffff)

    {

//	fprintf(stderr, "page not allocated [%x - %x:%x]\n",

//	    sector_num, extent_index, extent_offset);

	return -1; // not allocated

    }



    bitmap_offset = s->data_offset + (512 * s->catalog_bitmap[extent_index] *

	(s->extent_blocks + s->bitmap_blocks));

    block_offset = bitmap_offset + (512 * (s->bitmap_blocks + extent_offset));



//    fprintf(stderr, "sect: %x [ext i: %x o: %x] -> %x bitmap: %x block: %x\n",

//	sector_num, extent_index, extent_offset,

//	le32_to_cpu(s->catalog_bitmap[extent_index]),

//	bitmap_offset, block_offset);



    // read in bitmap for current extent

    lseek(s->fd, bitmap_offset + (extent_offset / 8), SEEK_SET);



    if (read(s->fd, &bitmap_entry, 1) != 1)

        return -1;



    if (!((bitmap_entry >> (extent_offset % 8)) & 1))

    {

//	fprintf(stderr, "sector (%x) in bitmap not allocated\n",

//	    sector_num);

	return -1; // not allocated

    }



    lseek(s->fd, block_offset, SEEK_SET);



    return 0;

}
