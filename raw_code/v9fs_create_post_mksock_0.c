static void v9fs_create_post_mksock(V9fsState *s, V9fsCreateState *vs,

                                                                int err)

{

    if (err) {

        err = -errno;

        goto out;

    }



    err = v9fs_do_chmod(s, &vs->fullname, vs->perm & 0777);

    v9fs_create_post_perms(s, vs, err);

    return;



out:

    v9fs_post_create(s, vs, err);

}
