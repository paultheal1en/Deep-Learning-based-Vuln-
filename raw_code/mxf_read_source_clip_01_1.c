static int mxf_read_source_clip(void *arg, AVIOContext *pb, int tag, int size, UID uid)

{

    MXFStructuralComponent *source_clip = arg;

    switch(tag) {

    case 0x0202:

        source_clip->duration = avio_rb64(pb);

        break;

    case 0x1201:

        source_clip->start_position = avio_rb64(pb);

        break;

    case 0x1101:

        /* UMID, only get last 16 bytes */

        avio_skip(pb, 16);

        avio_read(pb, source_clip->source_package_uid, 16);

        break;

    case 0x1102:

        source_clip->source_track_id = avio_rb32(pb);

        break;

    }

    return 0;

}
