static void nand_realize(DeviceState *dev, Error **errp)

{

    int pagesize;

    NANDFlashState *s = NAND(dev);



    s->buswidth = nand_flash_ids[s->chip_id].width >> 3;

    s->size = nand_flash_ids[s->chip_id].size << 20;

    if (nand_flash_ids[s->chip_id].options & NAND_SAMSUNG_LP) {

        s->page_shift = 11;

        s->erase_shift = 6;

    } else {

        s->page_shift = nand_flash_ids[s->chip_id].page_shift;

        s->erase_shift = nand_flash_ids[s->chip_id].erase_shift;

    }



    switch (1 << s->page_shift) {

    case 256:

        nand_init_256(s);

        break;

    case 512:

        nand_init_512(s);

        break;

    case 2048:

        nand_init_2048(s);

        break;

    default:

        error_setg(errp, "Unsupported NAND block size %#x\n",

                   1 << s->page_shift);

        return;

    }



    pagesize = 1 << s->oob_shift;

    s->mem_oob = 1;

    if (s->bdrv) {

        if (bdrv_is_read_only(s->bdrv)) {

            error_setg(errp, "Can't use a read-only drive");

            return;

        }

        if (bdrv_getlength(s->bdrv) >=

                (s->pages << s->page_shift) + (s->pages << s->oob_shift)) {

            pagesize = 0;

            s->mem_oob = 0;

        }

    } else {

        pagesize += 1 << s->page_shift;

    }

    if (pagesize) {

        s->storage = (uint8_t *) memset(g_malloc(s->pages * pagesize),

                        0xff, s->pages * pagesize);

    }

    /* Give s->ioaddr a sane value in case we save state before it is used. */

    s->ioaddr = s->io;

}
