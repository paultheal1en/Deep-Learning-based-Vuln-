static int64_t coroutine_fn vpc_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum)

{

    BDRVVPCState *s = bs->opaque;

    VHDFooter *footer = (VHDFooter*) s->footer_buf;

    int64_t start, offset;

    bool allocated;

    int n;



    if (be32_to_cpu(footer->type) == VHD_FIXED) {

        *pnum = nb_sectors;

        return BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID | BDRV_BLOCK_DATA |

               (sector_num << BDRV_SECTOR_BITS);

    }



    offset = get_sector_offset(bs, sector_num, 0);

    start = offset;

    allocated = (offset != -1);

    *pnum = 0;



    do {

        /* All sectors in a block are contiguous (without using the bitmap) */

        n = ROUND_UP(sector_num + 1, s->block_size / BDRV_SECTOR_SIZE)

          - sector_num;

        n = MIN(n, nb_sectors);



        *pnum += n;

        sector_num += n;

        nb_sectors -= n;

        /* *pnum can't be greater than one block for allocated

         * sectors since there is always a bitmap in between. */

        if (allocated) {

            return BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | start;

        }

        if (nb_sectors == 0) {

            break;

        }

        offset = get_sector_offset(bs, sector_num, 0);

    } while (offset == -1);



    return 0;

}
