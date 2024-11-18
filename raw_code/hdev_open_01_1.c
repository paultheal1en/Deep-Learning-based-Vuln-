static int hdev_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVRawState *s = bs->opaque;

    Error *local_err = NULL;

    int ret;



#if defined(__APPLE__) && defined(__MACH__)







    const char *filename = qdict_get_str(options, "filename");

    char bsd_path[MAXPATHLEN] = "";

    bool error_occurred = false;



    /* If using a real cdrom */

    if (strcmp(filename, "/dev/cdrom") == 0) {

        char *mediaType = NULL;

        kern_return_t ret_val;

        io_iterator_t mediaIterator = 0;



        mediaType = FindEjectableOpticalMedia(&mediaIterator);

        if (mediaType == NULL) {

            error_setg(errp, "Please make sure your CD/DVD is in the optical"

                       " drive");

            error_occurred = true;

            goto hdev_open_Mac_error;

        }



        ret_val = GetBSDPath(mediaIterator, bsd_path, sizeof(bsd_path), flags);

        if (ret_val != KERN_SUCCESS) {

            error_setg(errp, "Could not get BSD path for optical drive");

            error_occurred = true;

            goto hdev_open_Mac_error;

        }



        /* If a real optical drive was not found */

        if (bsd_path[0] == '\0') {

            error_setg(errp, "Failed to obtain bsd path for optical drive");

            error_occurred = true;

            goto hdev_open_Mac_error;

        }



        /* If using a cdrom disc and finding a partition on the disc failed */

        if (strncmp(mediaType, kIOCDMediaClass, 9) == 0 &&

            setup_cdrom(bsd_path, errp) == false) {

            print_unmounting_directions(bsd_path);

            error_occurred = true;

            goto hdev_open_Mac_error;

        }



        qdict_put(options, "filename", qstring_from_str(bsd_path));



hdev_open_Mac_error:

        g_free(mediaType);

        if (mediaIterator) {

            IOObjectRelease(mediaIterator);

        }

        if (error_occurred) {

            return -ENOENT;

        }

    }

#endif /* defined(__APPLE__) && defined(__MACH__) */



    s->type = FTYPE_FILE;



    ret = raw_open_common(bs, options, flags, 0, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

#if defined(__APPLE__) && defined(__MACH__)

        if (*bsd_path) {

            filename = bsd_path;

        }

        /* if a physical device experienced an error while being opened */

        if (strncmp(filename, "/dev/", 5) == 0) {

            print_unmounting_directions(filename);

        }

#endif /* defined(__APPLE__) && defined(__MACH__) */

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