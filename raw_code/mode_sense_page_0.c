static int mode_sense_page(SCSIDiskState *s, int page, uint8_t **p_outbuf,

                           int page_control)

{

    static const int mode_sense_valid[0x3f] = {

        [MODE_PAGE_HD_GEOMETRY]            = (1 << TYPE_DISK),

        [MODE_PAGE_FLEXIBLE_DISK_GEOMETRY] = (1 << TYPE_DISK),

        [MODE_PAGE_CACHING]                = (1 << TYPE_DISK) | (1 << TYPE_ROM),

        [MODE_PAGE_CAPABILITIES]           = (1 << TYPE_ROM),

    };



    BlockDriverState *bdrv = s->bs;

    int cylinders, heads, secs;

    uint8_t *p = *p_outbuf;



    if ((mode_sense_valid[page] & (1 << s->qdev.type)) == 0) {

        return -1;

    }



    p[0] = page;



    /*

     * If Changeable Values are requested, a mask denoting those mode parameters

     * that are changeable shall be returned. As we currently don't support

     * parameter changes via MODE_SELECT all bits are returned set to zero.

     * The buffer was already menset to zero by the caller of this function.

     */

    switch (page) {

    case MODE_PAGE_HD_GEOMETRY:

        p[1] = 0x16;

        if (page_control == 1) { /* Changeable Values */

            break;

        }

        /* if a geometry hint is available, use it */

        bdrv_get_geometry_hint(bdrv, &cylinders, &heads, &secs);

        p[2] = (cylinders >> 16) & 0xff;

        p[3] = (cylinders >> 8) & 0xff;

        p[4] = cylinders & 0xff;

        p[5] = heads & 0xff;

        /* Write precomp start cylinder, disabled */

        p[6] = (cylinders >> 16) & 0xff;

        p[7] = (cylinders >> 8) & 0xff;

        p[8] = cylinders & 0xff;

        /* Reduced current start cylinder, disabled */

        p[9] = (cylinders >> 16) & 0xff;

        p[10] = (cylinders >> 8) & 0xff;

        p[11] = cylinders & 0xff;

        /* Device step rate [ns], 200ns */

        p[12] = 0;

        p[13] = 200;

        /* Landing zone cylinder */

        p[14] = 0xff;

        p[15] =  0xff;

        p[16] = 0xff;

        /* Medium rotation rate [rpm], 5400 rpm */

        p[20] = (5400 >> 8) & 0xff;

        p[21] = 5400 & 0xff;

        break;



    case MODE_PAGE_FLEXIBLE_DISK_GEOMETRY:

        p[1] = 0x1e;

        if (page_control == 1) { /* Changeable Values */

            break;

        }

        /* Transfer rate [kbit/s], 5Mbit/s */

        p[2] = 5000 >> 8;

        p[3] = 5000 & 0xff;

        /* if a geometry hint is available, use it */

        bdrv_get_geometry_hint(bdrv, &cylinders, &heads, &secs);

        p[4] = heads & 0xff;

        p[5] = secs & 0xff;

        p[6] = s->cluster_size * 2;

        p[8] = (cylinders >> 8) & 0xff;

        p[9] = cylinders & 0xff;

        /* Write precomp start cylinder, disabled */

        p[10] = (cylinders >> 8) & 0xff;

        p[11] = cylinders & 0xff;

        /* Reduced current start cylinder, disabled */

        p[12] = (cylinders >> 8) & 0xff;

        p[13] = cylinders & 0xff;

        /* Device step rate [100us], 100us */

        p[14] = 0;

        p[15] = 1;

        /* Device step pulse width [us], 1us */

        p[16] = 1;

        /* Device head settle delay [100us], 100us */

        p[17] = 0;

        p[18] = 1;

        /* Motor on delay [0.1s], 0.1s */

        p[19] = 1;

        /* Motor off delay [0.1s], 0.1s */

        p[20] = 1;

        /* Medium rotation rate [rpm], 5400 rpm */

        p[28] = (5400 >> 8) & 0xff;

        p[29] = 5400 & 0xff;

        break;



    case MODE_PAGE_CACHING:

        p[0] = 8;

        p[1] = 0x12;

        if (page_control == 1) { /* Changeable Values */

            break;

        }

        if (bdrv_enable_write_cache(s->bs)) {

            p[2] = 4; /* WCE */

        }

        break;



    case MODE_PAGE_CAPABILITIES:

        p[1] = 0x14;

        if (page_control == 1) { /* Changeable Values */

            break;

        }

        p[2] = 3; // CD-R & CD-RW read

        p[3] = 0; // Writing not supported

        p[4] = 0x7f; /* Audio, composite, digital out,

                        mode 2 form 1&2, multi session */

        p[5] = 0xff; /* CD DA, DA accurate, RW supported,

                        RW corrected, C2 errors, ISRC,

                        UPC, Bar code */

        p[6] = 0x2d | (s->tray_locked ? 2 : 0);

        /* Locking supported, jumper present, eject, tray */

        p[7] = 0; /* no volume & mute control, no

                     changer */

        p[8] = (50 * 176) >> 8; // 50x read speed

        p[9] = (50 * 176) & 0xff;

        p[10] = 0 >> 8; // No volume

        p[11] = 0 & 0xff;

        p[12] = 2048 >> 8; // 2M buffer

        p[13] = 2048 & 0xff;

        p[14] = (16 * 176) >> 8; // 16x read speed current

        p[15] = (16 * 176) & 0xff;

        p[18] = (16 * 176) >> 8; // 16x write speed

        p[19] = (16 * 176) & 0xff;

        p[20] = (16 * 176) >> 8; // 16x write speed current

        p[21] = (16 * 176) & 0xff;

        break;



    default:

        return -1;

    }



    *p_outbuf += p[1] + 2;

    return p[1] + 2;

}
