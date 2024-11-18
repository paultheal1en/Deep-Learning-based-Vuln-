static int local_set_xattr(const char *path, FsCred *credp)

{

    int err;



    if (credp->fc_uid != -1) {

        err = setxattr(path, "user.virtfs.uid", &credp->fc_uid, sizeof(uid_t),

                0);

        if (err) {

            return err;

        }

    }

    if (credp->fc_gid != -1) {

        err = setxattr(path, "user.virtfs.gid", &credp->fc_gid, sizeof(gid_t),

                0);

        if (err) {

            return err;

        }

    }

    if (credp->fc_mode != -1) {

        err = setxattr(path, "user.virtfs.mode", &credp->fc_mode,

                sizeof(mode_t), 0);

        if (err) {

            return err;

        }

    }

    if (credp->fc_rdev != -1) {

        err = setxattr(path, "user.virtfs.rdev", &credp->fc_rdev,

                sizeof(dev_t), 0);

        if (err) {

            return err;

        }

    }

    return 0;

}
