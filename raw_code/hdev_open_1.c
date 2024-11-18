static int hdev_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVRawState *s = bs->opaque;



#if defined(__APPLE__) && defined(__MACH__)

    if (strstart(filename, "/dev/cdrom", NULL)) {

        kern_return_t kernResult;

        io_iterator_t mediaIterator;

        char bsdPath[ MAXPATHLEN ];

        int fd;



        kernResult = FindEjectableCDMedia( &mediaIterator );

        kernResult = GetBSDPath( mediaIterator, bsdPath, sizeof( bsdPath ) );



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

        }



        if ( mediaIterator )

            IOObjectRelease( mediaIterator );

    }

#endif



    s->type = FTYPE_FILE;

#if defined(__linux__)

    {

        char resolved_path[ MAXPATHLEN ], *temp;



        temp = realpath(filename, resolved_path);

        if (temp && strstart(temp, "/dev/sg", NULL)) {

            bs->sg = 1;

        }

    }

#endif



    return raw_open_common(bs, filename, flags, 0);

}
