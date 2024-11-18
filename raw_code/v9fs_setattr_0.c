static void v9fs_setattr(void *opaque)

{

    int err = 0;

    int32_t fid;

    V9fsFidState *fidp;

    size_t offset = 7;

    V9fsIattr v9iattr;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "dI", &fid, &v9iattr);



    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -EINVAL;

        goto out_nofid;

    }

    if (v9iattr.valid & P9_ATTR_MODE) {

        err = v9fs_co_chmod(pdu, &fidp->path, v9iattr.mode);

        if (err < 0) {

            goto out;

        }

    }

    if (v9iattr.valid & (P9_ATTR_ATIME | P9_ATTR_MTIME)) {

        struct timespec times[2];

        if (v9iattr.valid & P9_ATTR_ATIME) {

            if (v9iattr.valid & P9_ATTR_ATIME_SET) {

                times[0].tv_sec = v9iattr.atime_sec;

                times[0].tv_nsec = v9iattr.atime_nsec;

            } else {

                times[0].tv_nsec = UTIME_NOW;

            }

        } else {

            times[0].tv_nsec = UTIME_OMIT;

        }

        if (v9iattr.valid & P9_ATTR_MTIME) {

            if (v9iattr.valid & P9_ATTR_MTIME_SET) {

                times[1].tv_sec = v9iattr.mtime_sec;

                times[1].tv_nsec = v9iattr.mtime_nsec;

            } else {

                times[1].tv_nsec = UTIME_NOW;

            }

        } else {

            times[1].tv_nsec = UTIME_OMIT;

        }

        err = v9fs_co_utimensat(pdu, &fidp->path, times);

        if (err < 0) {

            goto out;

        }

    }

    /*

     * If the only valid entry in iattr is ctime we can call

     * chown(-1,-1) to update the ctime of the file

     */

    if ((v9iattr.valid & (P9_ATTR_UID | P9_ATTR_GID)) ||

        ((v9iattr.valid & P9_ATTR_CTIME)

         && !((v9iattr.valid & P9_ATTR_MASK) & ~P9_ATTR_CTIME))) {

        if (!(v9iattr.valid & P9_ATTR_UID)) {

            v9iattr.uid = -1;

        }

        if (!(v9iattr.valid & P9_ATTR_GID)) {

            v9iattr.gid = -1;

        }

        err = v9fs_co_chown(pdu, &fidp->path, v9iattr.uid,

                            v9iattr.gid);

        if (err < 0) {

            goto out;

        }

    }

    if (v9iattr.valid & (P9_ATTR_SIZE)) {

        err = v9fs_co_truncate(pdu, &fidp->path, v9iattr.size);

        if (err < 0) {

            goto out;

        }

    }

    err = offset;

out:

    put_fid(pdu, fidp);

out_nofid:

    complete_pdu(s, pdu, err);

}
