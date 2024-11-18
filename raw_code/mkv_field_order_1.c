static int mkv_field_order(MatroskaDemuxContext *matroska, int64_t field_order)

{

    int major, minor, micro, bttb = 0;



    /* workaround a bug in our Matroska muxer, introduced in version 57.36 alongside

     * this function, and fixed in 57.52 */

    if (sscanf(matroska->muxingapp, "Lavf%d.%d.%d", &major, &minor, &micro) == 3)

        bttb = (major == 57 && minor >= 36 && minor <= 51 && micro >= 100);



    switch (field_order) {

    case MATROSKA_VIDEO_FIELDORDER_PROGRESSIVE:

        return AV_FIELD_PROGRESSIVE;

    case MATROSKA_VIDEO_FIELDORDER_UNDETERMINED:

        return AV_FIELD_UNKNOWN;

    case MATROSKA_VIDEO_FIELDORDER_TT:

        return AV_FIELD_TT;

    case MATROSKA_VIDEO_FIELDORDER_BB:

        return AV_FIELD_BB;

    case MATROSKA_VIDEO_FIELDORDER_BT:

        return bttb ? AV_FIELD_TB : AV_FIELD_BT;

    case MATROSKA_VIDEO_FIELDORDER_TB:

        return bttb ? AV_FIELD_BT : AV_FIELD_TB;

    default:

        return AV_FIELD_UNKNOWN;

    }

}
