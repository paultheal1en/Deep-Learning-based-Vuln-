check_host_key_hash(BDRVSSHState *s, const char *hash,

                    int hash_type, size_t fingerprint_len)

{

    const char *fingerprint;



    fingerprint = libssh2_hostkey_hash(s->session, hash_type);

    if (!fingerprint) {

        session_error_report(s, "failed to read remote host key");

        return -EINVAL;

    }



    if(compare_fingerprint((unsigned char *) fingerprint, fingerprint_len,

                           hash) != 0) {

        error_report("remote host key does not match host_key_check '%s'",

                     hash);

        return -EPERM;

    }



    return 0;

}
