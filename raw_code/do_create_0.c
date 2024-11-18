static int do_create(struct iovec *iovec)

{

    int ret;

    V9fsString path;

    int flags, mode, uid, gid, cur_uid, cur_gid;



    v9fs_string_init(&path);

    ret = proxy_unmarshal(iovec, PROXY_HDR_SZ, "sdddd",

                          &path, &flags, &mode, &uid, &gid);

    if (ret < 0) {

        goto unmarshal_err_out;

    }

    cur_uid = geteuid();

    cur_gid = getegid();

    ret = setfsugid(uid, gid);

    if (ret < 0) {

        /*

         * On failure reset back to the

         * old uid/gid

         */

        ret = -errno;

        goto err_out;

    }

    ret = open(path.data, flags, mode);

    if (ret < 0) {

        ret = -errno;

    }



err_out:

    setfsugid(cur_uid, cur_gid);

unmarshal_err_out:

    v9fs_string_free(&path);

    return ret;

}
