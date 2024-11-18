int v9fs_co_symlink(V9fsState *s, V9fsFidState *fidp,

                    const char *oldpath, const char *newpath, gid_t gid)

{

    int err;

    FsCred cred;



    cred_init(&cred);

    cred.fc_uid = fidp->uid;

    cred.fc_gid = gid;

    cred.fc_mode = 0777;

    v9fs_co_run_in_worker(

        {

            err = s->ops->symlink(&s->ctx, oldpath, newpath, &cred);

            if (err < 0) {

                err = -errno;

            }

        });

    return err;

}
