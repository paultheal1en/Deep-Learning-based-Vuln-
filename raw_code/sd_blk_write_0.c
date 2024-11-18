static void sd_blk_write(SDState *sd, uint64_t addr, uint32_t len)

{

    uint64_t end = addr + len;



    if ((addr & 511) || len < 512)

        if (!sd->bdrv || bdrv_read(sd->bdrv, addr >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_write: read error on host side\n");

            return;

        }



    if (end > (addr & ~511) + 512) {

        memcpy(sd->buf + (addr & 511), sd->data, 512 - (addr & 511));

        if (bdrv_write(sd->bdrv, addr >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_write: write error on host side\n");

            return;

        }



        if (bdrv_read(sd->bdrv, end >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_write: read error on host side\n");

            return;

        }

        memcpy(sd->buf, sd->data + 512 - (addr & 511), end & 511);

        if (bdrv_write(sd->bdrv, end >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_write: write error on host side\n");

        }

    } else {

        memcpy(sd->buf + (addr & 511), sd->data, len);

        if (!sd->bdrv || bdrv_write(sd->bdrv, addr >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_write: write error on host side\n");

        }

    }

}
