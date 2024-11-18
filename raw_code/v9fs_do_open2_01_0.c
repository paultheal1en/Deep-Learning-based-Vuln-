static int v9fs_do_open2(V9fsState *s, V9fsCreateState *vs)

{

    FsCred cred;

    int flags;



    cred_init(&cred);

    cred.fc_uid = vs->fidp->uid;

    cred.fc_mode = vs->perm & 0777;

    flags = omode_to_uflags(vs->mode) | O_CREAT;



    return s->ops->open2(&s->ctx, vs->fullname.data, flags, &cred);

}
