static int mxf_read_source_package(MXFPackage *package, ByteIOContext *pb, int tag)

{

    switch(tag) {

    case 0x4403:

        package->tracks_count = get_be32(pb);

        if (package->tracks_count >= UINT_MAX / sizeof(UID))

            return -1;

        package->tracks_refs = av_malloc(package->tracks_count * sizeof(UID));

        if (!package->tracks_refs)

            return -1;

        url_fskip(pb, 4); /* useless size of objects, always 16 according to specs */

        get_buffer(pb, (uint8_t *)package->tracks_refs, package->tracks_count * sizeof(UID));

        break;

    case 0x4401:

        /* UMID, only get last 16 bytes */

        url_fskip(pb, 16);

        get_buffer(pb, package->package_uid, 16);

        break;

    case 0x4701:

        get_buffer(pb, package->descriptor_ref, 16);

        break;

    }

    return 0;

}
