int qcrypto_cipher_encrypt(QCryptoCipher *cipher,
                           const void *in,
                           void *out,
                           size_t len,
                           Error **errp)
{
    QCryptoCipherBuiltin *ctxt = cipher->opaque;
    return ctxt->encrypt(cipher, in, out, len, errp);