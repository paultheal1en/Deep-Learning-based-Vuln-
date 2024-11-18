static int io_write_data_type(void *opaque, uint8_t *buf, int size,

                              enum AVIODataMarkerType type, int64_t time)

{

    char timebuf[30], content[5] = { 0 };

    const char *str;

    switch (type) {

    case AVIO_DATA_MARKER_HEADER:         str = "header";   break;

    case AVIO_DATA_MARKER_SYNC_POINT:     str = "sync";     break;

    case AVIO_DATA_MARKER_BOUNDARY_POINT: str = "boundary"; break;

    case AVIO_DATA_MARKER_UNKNOWN:        str = "unknown";  break;

    case AVIO_DATA_MARKER_TRAILER:        str = "trailer";  break;


    }

    if (time == AV_NOPTS_VALUE)

        snprintf(timebuf, sizeof(timebuf), "nopts");

    else

        snprintf(timebuf, sizeof(timebuf), "%"PRId64, time);

    // There can be multiple header/trailer callbacks, only log the box type

    // for header at out_size == 0

    if (type != AVIO_DATA_MARKER_UNKNOWN &&

        type != AVIO_DATA_MARKER_TRAILER &&

        (type != AVIO_DATA_MARKER_HEADER || out_size == 0) &&

        size >= 8)

        memcpy(content, &buf[4], 4);

    else

        snprintf(content, sizeof(content), "-");

    printf("write_data len %d, time %s, type %s atom %s\n", size, timebuf, str, content);

    return io_write(opaque, buf, size);

}