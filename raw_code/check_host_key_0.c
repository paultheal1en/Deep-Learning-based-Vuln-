static int check_host_key(BDRVSSHState *s, const char *host, int port,

                          const char *host_key_check)

{

    /* host_key_check=no */

    if (strcmp(host_key_check, "no") == 0) {

        return 0;

    }



    /* host_key_check=md5:xx:yy:zz:... */

    if (strncmp(host_key_check, "md5:", 4) == 0) {

        return check_host_key_hash(s, &host_key_check[4],

                                   LIBSSH2_HOSTKEY_HASH_MD5, 16);

    }



    /* host_key_check=sha1:xx:yy:zz:... */

    if (strncmp(host_key_check, "sha1:", 5) == 0) {

        return check_host_key_hash(s, &host_key_check[5],

                                   LIBSSH2_HOSTKEY_HASH_SHA1, 20);

    }



    /* host_key_check=yes */

    if (strcmp(host_key_check, "yes") == 0) {

        return check_host_key_knownhosts(s, host, port);

    }



    error_report("unknown host_key_check setting (%s)", host_key_check);

    return -EINVAL;

}
