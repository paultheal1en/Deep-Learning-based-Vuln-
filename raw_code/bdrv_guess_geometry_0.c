void bdrv_guess_geometry(BlockDriverState *bs, int *pcyls, int *pheads, int *psecs)

{

    int translation, lba_detected = 0;

    int cylinders, heads, secs;

    uint64_t nb_sectors;



    /* if a geometry hint is available, use it */

    bdrv_get_geometry(bs, &nb_sectors);

    bdrv_get_geometry_hint(bs, &cylinders, &heads, &secs);

    translation = bdrv_get_translation_hint(bs);

    if (cylinders != 0) {

        *pcyls = cylinders;

        *pheads = heads;

        *psecs = secs;

    } else {

        if (guess_disk_lchs(bs, &cylinders, &heads, &secs) == 0) {

            if (heads > 16) {

                /* if heads > 16, it means that a BIOS LBA

                   translation was active, so the default

                   hardware geometry is OK */

                lba_detected = 1;

                goto default_geometry;

            } else {

                *pcyls = cylinders;

                *pheads = heads;

                *psecs = secs;

                /* disable any translation to be in sync with

                   the logical geometry */

                if (translation == BIOS_ATA_TRANSLATION_AUTO) {

                    bdrv_set_translation_hint(bs,

                                              BIOS_ATA_TRANSLATION_NONE);

                }

            }

        } else {

        default_geometry:

            /* if no geometry, use a standard physical disk geometry */

            cylinders = nb_sectors / (16 * 63);



            if (cylinders > 16383)

                cylinders = 16383;

            else if (cylinders < 2)

                cylinders = 2;

            *pcyls = cylinders;

            *pheads = 16;

            *psecs = 63;

            if ((lba_detected == 1) && (translation == BIOS_ATA_TRANSLATION_AUTO)) {

                if ((*pcyls * *pheads) <= 131072) {

                    bdrv_set_translation_hint(bs,

                                              BIOS_ATA_TRANSLATION_LARGE);

                } else {

                    bdrv_set_translation_hint(bs,

                                              BIOS_ATA_TRANSLATION_LBA);

                }

            }

        }

        bdrv_set_geometry_hint(bs, *pcyls, *pheads, *psecs);

    }

}
