static int cryptodev_builtin_create_cipher_session(

                    CryptoDevBackendBuiltin *builtin,

                    CryptoDevBackendSymSessionInfo *sess_info,

                    Error **errp)

{

    int algo;

    int mode;

    QCryptoCipher *cipher;

    int index;

    CryptoDevBackendBuiltinSession *sess;



    if (sess_info->op_type != VIRTIO_CRYPTO_SYM_OP_CIPHER) {

        error_setg(errp, "Unsupported optype :%u", sess_info->op_type);

        return -1;

    }



    index = cryptodev_builtin_get_unused_session_index(builtin);

    if (index < 0) {

        error_setg(errp, "Total number of sessions created exceeds %u",

                  MAX_NUM_SESSIONS);

        return -1;

    }



    switch (sess_info->cipher_alg) {

    case VIRTIO_CRYPTO_CIPHER_AES_ECB:

        algo = cryptodev_builtin_get_aes_algo(sess_info->key_len,

                                                          errp);

        if (algo < 0)  {

            return -1;

        }

        mode = QCRYPTO_CIPHER_MODE_ECB;

        break;

    case VIRTIO_CRYPTO_CIPHER_AES_CBC:

        algo = cryptodev_builtin_get_aes_algo(sess_info->key_len,

                                                          errp);

        if (algo < 0)  {

            return -1;

        }

        mode = QCRYPTO_CIPHER_MODE_CBC;

        break;

    case VIRTIO_CRYPTO_CIPHER_AES_CTR:

        algo = cryptodev_builtin_get_aes_algo(sess_info->key_len,

                                                          errp);

        if (algo < 0)  {

            return -1;

        }

        mode = QCRYPTO_CIPHER_MODE_CTR;

        break;

    case VIRTIO_CRYPTO_CIPHER_DES_ECB:

        algo = QCRYPTO_CIPHER_ALG_DES_RFB;

        mode = QCRYPTO_CIPHER_MODE_ECB;

        break;

    default:

        error_setg(errp, "Unsupported cipher alg :%u",

                   sess_info->cipher_alg);

        return -1;

    }



    cipher = qcrypto_cipher_new(algo, mode,

                               sess_info->cipher_key,

                               sess_info->key_len,

                               errp);

    if (!cipher) {

        return -1;

    }



    sess = g_new0(CryptoDevBackendBuiltinSession, 1);

    sess->cipher = cipher;

    sess->direction = sess_info->direction;

    sess->type = sess_info->op_type;



    builtin->sessions[index] = sess;



    return index;

}
