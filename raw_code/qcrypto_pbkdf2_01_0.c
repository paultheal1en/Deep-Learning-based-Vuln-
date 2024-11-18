int qcrypto_pbkdf2(QCryptoHashAlgorithm hash G_GNUC_UNUSED,

                   const uint8_t *key G_GNUC_UNUSED,

                   size_t nkey G_GNUC_UNUSED,

                   const uint8_t *salt G_GNUC_UNUSED,

                   size_t nsalt G_GNUC_UNUSED,

                   unsigned int iterations G_GNUC_UNUSED,

                   uint8_t *out G_GNUC_UNUSED,

                   size_t nout G_GNUC_UNUSED,

                   Error **errp)

{

    error_setg_errno(errp, ENOSYS,

                     "No crypto library supporting PBKDF in this build");

    return -1;

}
