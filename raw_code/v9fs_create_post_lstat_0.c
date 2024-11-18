static void v9fs_create_post_lstat(V9fsState *s, V9fsCreateState *vs, int err)

{



    if (err == 0 || errno != ENOENT) {

        err = -errno;

        goto out;

    }



    if (vs->perm & P9_STAT_MODE_DIR) {

        err = v9fs_do_mkdir(s, vs);

        v9fs_create_post_mkdir(s, vs, err);

    } else if (vs->perm & P9_STAT_MODE_SYMLINK) {

        err = v9fs_do_symlink(s, vs);

        v9fs_create_post_perms(s, vs, err);

    } else if (vs->perm & P9_STAT_MODE_LINK) {

        int32_t nfid = atoi(vs->extension.data);

        V9fsFidState *nfidp = lookup_fid(s, nfid);

        if (nfidp == NULL) {

            err = -errno;

            v9fs_post_create(s, vs, err);

        }

        err = v9fs_do_link(s, &nfidp->path, &vs->fullname);

        v9fs_create_post_perms(s, vs, err);

    } else if (vs->perm & P9_STAT_MODE_DEVICE) {

        char ctype;

        uint32_t major, minor;

        mode_t nmode = 0;



        if (sscanf(vs->extension.data, "%c %u %u", &ctype, &major,

                                        &minor) != 3) {

            err = -errno;

            v9fs_post_create(s, vs, err);

        }



        switch (ctype) {

        case 'c':

            nmode = S_IFCHR;

            break;

        case 'b':

            nmode = S_IFBLK;

            break;

        default:

            err = -EIO;

            v9fs_post_create(s, vs, err);

        }



        nmode |= vs->perm & 0777;

        err = v9fs_do_mknod(s, vs, nmode, makedev(major, minor));

        v9fs_create_post_perms(s, vs, err);

    } else if (vs->perm & P9_STAT_MODE_NAMED_PIPE) {

        err = v9fs_do_mknod(s, vs, S_IFIFO | (vs->perm & 0777), 0);

        v9fs_post_create(s, vs, err);

    } else if (vs->perm & P9_STAT_MODE_SOCKET) {

        err = v9fs_do_mksock(s, &vs->fullname);

        v9fs_create_post_mksock(s, vs, err);

    } else {

        vs->fidp->fd = v9fs_do_open2(s, vs);

        v9fs_create_post_open2(s, vs, err);

    }



    return;



out:

    v9fs_post_create(s, vs, err);

}
