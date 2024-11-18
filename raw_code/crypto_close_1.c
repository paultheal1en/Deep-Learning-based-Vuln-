static int crypto_close(URLContext *h)

{

    CryptoContext *c = h->priv_data;

    if (c->hd)

        ffurl_close(c->hd);

    av_freep(&c->aes);

    av_freep(&c->key);

    av_freep(&c->iv);

    return 0;

}
