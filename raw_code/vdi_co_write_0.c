static int vdi_co_write(BlockDriverState *bs,

        int64_t sector_num, const uint8_t *buf, int nb_sectors)

{

    BDRVVdiState *s = bs->opaque;

    uint32_t bmap_entry;

    uint32_t block_index;

    uint32_t sector_in_block;

    uint32_t n_sectors;

    uint32_t bmap_first = VDI_UNALLOCATED;

    uint32_t bmap_last = VDI_UNALLOCATED;

    uint8_t *block = NULL;

    int ret;



    logout("\n");



restart:

    block_index = sector_num / s->block_sectors;

    sector_in_block = sector_num % s->block_sectors;

    n_sectors = s->block_sectors - sector_in_block;

    if (n_sectors > nb_sectors) {

        n_sectors = nb_sectors;

    }



    logout("will write %u sectors starting at sector %" PRIu64 "\n",

           n_sectors, sector_num);



    /* prepare next AIO request */

    bmap_entry = le32_to_cpu(s->bmap[block_index]);

    if (!VDI_IS_ALLOCATED(bmap_entry)) {

        /* Allocate new block and write to it. */

        uint64_t offset;

        bmap_entry = s->header.blocks_allocated;

        s->bmap[block_index] = cpu_to_le32(bmap_entry);

        s->header.blocks_allocated++;

        offset = s->header.offset_data / SECTOR_SIZE +

                 (uint64_t)bmap_entry * s->block_sectors;

        if (block == NULL) {

            block = g_malloc(s->block_size);

            bmap_first = block_index;

        }

        bmap_last = block_index;

        /* Copy data to be written to new block and zero unused parts. */

        memset(block, 0, sector_in_block * SECTOR_SIZE);

        memcpy(block + sector_in_block * SECTOR_SIZE,

               buf, n_sectors * SECTOR_SIZE);

        memset(block + (sector_in_block + n_sectors) * SECTOR_SIZE, 0,

               (s->block_sectors - n_sectors - sector_in_block) * SECTOR_SIZE);

        ret = bdrv_write(bs->file, offset, block, s->block_sectors);

    } else {

        uint64_t offset = s->header.offset_data / SECTOR_SIZE +

                          (uint64_t)bmap_entry * s->block_sectors +

                          sector_in_block;

        ret = bdrv_write(bs->file, offset, buf, n_sectors);

    }



    nb_sectors -= n_sectors;

    sector_num += n_sectors;

    buf += n_sectors * SECTOR_SIZE;



    logout("%u sectors written\n", n_sectors);

    if (ret >= 0 && nb_sectors > 0) {

        goto restart;

    }



    logout("finished data write\n");

    if (ret < 0) {

        return ret;

    }



    if (block) {

        /* One or more new blocks were allocated. */

        VdiHeader *header = (VdiHeader *) block;

        uint8_t *base;

        uint64_t offset;



        logout("now writing modified header\n");

        assert(VDI_IS_ALLOCATED(bmap_first));

        *header = s->header;

        vdi_header_to_le(header);

        ret = bdrv_write(bs->file, 0, block, 1);

        g_free(block);

        block = NULL;



        if (ret < 0) {

            return ret;

        }



        logout("now writing modified block map entry %u...%u\n",

               bmap_first, bmap_last);

        /* Write modified sectors from block map. */

        bmap_first /= (SECTOR_SIZE / sizeof(uint32_t));

        bmap_last /= (SECTOR_SIZE / sizeof(uint32_t));

        n_sectors = bmap_last - bmap_first + 1;

        offset = s->bmap_sector + bmap_first;

        base = ((uint8_t *)&s->bmap[0]) + bmap_first * SECTOR_SIZE;

        logout("will write %u block map sectors starting from entry %u\n",

               n_sectors, bmap_first);

        ret = bdrv_write(bs->file, offset, base, n_sectors);

    }



    return ret;

}
