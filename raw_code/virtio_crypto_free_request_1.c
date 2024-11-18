static void virtio_crypto_free_request(VirtIOCryptoReq *req)

{

    if (req) {

        if (req->flags == CRYPTODEV_BACKEND_ALG_SYM) {

            g_free(req->u.sym_op_info);

        }

        g_free(req);

    }

}
