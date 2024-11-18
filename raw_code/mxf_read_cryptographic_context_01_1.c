static int mxf_read_cryptographic_context(MXFCryptoContext *cryptocontext, ByteIOContext *pb, int tag, int size, UID uid)

{

    if (size != 16)

        return -1;

    if (IS_KLV_KEY(uid, mxf_crypto_source_container_ul))

        get_buffer(pb, cryptocontext->source_container_ul, 16);

    return 0;

}
