static int hdev_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVRawState *s = bs->opaque;

    Error *local_err = NULL;

    int ret;



#if defined(__APPLE__) && defined(__MACH__)

    const char *filename = qdict_get_str(options, "filename");



    if (strstart(filename, "/dev/cdrom", NULL)) {

        kern_return_t kernResult;

        io_iterator_t mediaIterator;

        char bsdPath[ MAXPATHLEN ];

        int fd;



        kernResult = FindEjectableCDMedia( &mediaIterator );

        kernResult = GetBSDPath(mediaIterator, bsdPath, sizeof(bsdPath),

                                flags);

        if ( bsdPath[ 0 ] != '\0' ) {

            strcat(bsdPath,"s0");

            /* some CDs don't have a partition 0 */

            fd = qemu_open(bsdPath, O_RDONLY | O_BINARY | O_LARGEFILE);

            if (fd < 0) {

                bsdPath[strlen(bsdPath)-1] = '1';

            } else {

                qemu_close(fd);

            }

            filename = bsdPath;

            qdict_put(options, "filename", qstring_from_str(filename));

        }



        if ( mediaIterator )

            IOObjectRelease( mediaIterator );

    }

#endif



    s->type = FTYPE_FILE;



    ret = raw_open_common(bs, options, flags, 0, &local_err);

    if (ret < 0) {

        if (local_err) {

            error_propagate(errp, local_err);

        }

        return ret;

    }



    /* Since this does ioctl the device must be already opened */

    bs->sg = hdev_is_sg(bs);



    if (flags & BDRV_O_RDWR) {

        ret = check_hdev_writable(s);

        if (ret < 0) {

            raw_close(bs);

            error_setg_errno(errp, -ret, "The device is not writable");

            return ret;

        }

    }



    return ret;

}
