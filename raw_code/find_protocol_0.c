static BlockDriver *find_protocol(const char *filename)

{

    BlockDriver *drv1;

    char protocol[128];

    int len;

    const char *p;



#ifdef _WIN32

    if (is_windows_drive(filename) ||

        is_windows_drive_prefix(filename))

        return bdrv_find_format("raw");

#endif

    p = strchr(filename, ':');

    if (!p)

        return bdrv_find_format("raw");

    len = p - filename;

    if (len > sizeof(protocol) - 1)

        len = sizeof(protocol) - 1;

    memcpy(protocol, filename, len);

    protocol[len] = '\0';

    QLIST_FOREACH(drv1, &bdrv_drivers, list) {

        if (drv1->protocol_name &&

            !strcmp(drv1->protocol_name, protocol)) {

            return drv1;

        }

    }

    return NULL;

}
